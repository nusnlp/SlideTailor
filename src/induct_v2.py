import json
import os
import shutil
from collections import defaultdict

from jinja2 import Template

from model_utils import get_cluster, get_image_embedding, images_cosine_similarity
from presentation import Presentation
from utils import Config, pexists, pjoin, tenacity
import re


class SlideInducter:
    """
    Stage I: Presentation Analysis.
    This stage is to analyze the presentation: cluster slides into different layouts, and extract content schema for each layout.
    """

    def __init__(
        self,
        vision_model,
        language_model,
        prs: Presentation,
        ppt_image_folder: str,
        template_image_folder: str,
        config: Config,
        image_models: list,
        model_identifier: str,
        concise: bool = False,
    ):
        """
        Initialize the SlideInducter.

        Args:
            prs (Presentation): The presentation object.
            ppt_image_folder (str): The folder containing PPT images.
            template_image_folder (str): The folder containing normalized slide images.
            config (Config): The configuration object.
            image_models (list): A list of image models.
        """
        self.prs = prs
        self.config = config
        self.ppt_image_folder = ppt_image_folder
        self.template_image_folder = template_image_folder
        assert (
            len(os.listdir(template_image_folder))
            == len(prs)
            == len(os.listdir(ppt_image_folder))
        )
        self.image_models = image_models
        self.slide_induction = defaultdict(lambda: defaultdict(list))
        self.output_dir = pjoin(config.RUN_DIR, "template_induct", model_identifier)
        self.split_cache = pjoin(self.output_dir, f"split_cache.json")
        self.induct_cache = pjoin(self.output_dir, f"induct_cache.json")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.vision_model = vision_model
        self.language_model = language_model
        
        self.concise = concise
        
    def layout_induct(self):
        """
        Perform layout induction for the presentation.
        """
        if pexists(self.induct_cache):
            return json.load(open(self.induct_cache))
        content_slides_index, functional_cluster = self.category_split()    
        for layout_name, cluster in functional_cluster.items():
            for slide_idx in cluster:
                content_type = self.prs.slides[slide_idx - 1].get_content_type()
                self.slide_induction[layout_name + ":" + content_type]["slides"].append(
                    slide_idx
                )   
        for layout_name, cluster in self.slide_induction.items():   
            cluster["template_id"] = cluster["slides"][-1]

        functional_keys = list(self.slide_induction.keys())
        function_slides_index = set()
        for layout_name, cluster in self.slide_induction.items():
            function_slides_index.update(cluster["slides"]) 
        used_slides_index = function_slides_index.union(content_slides_index)   
        for i in range(len(self.prs.slides)):
            if i + 1 not in used_slides_index:
                content_slides_index.add(i + 1)
        self.layout_split(content_slides_index) 
        if self.config.DEBUG:
            for layout_name, cluster in self.slide_induction.items():   
                cluster_dir = pjoin(self.output_dir, "cluster_slides", layout_name)
                os.makedirs(cluster_dir, exist_ok=True)
                for slide_idx in cluster["slides"]:
                    shutil.copy(
                        pjoin(self.ppt_image_folder, f"slide_{slide_idx:04d}.jpg"),
                        pjoin(cluster_dir, f"slide_{slide_idx:04d}.jpg"),
                    )
        self.slide_induction["functional_keys"] = functional_keys
        json.dump(
            self.slide_induction,
            open(self.induct_cache, "w"),
            indent=4,
            ensure_ascii=False,
        )
        return self.slide_induction
    

    def category_split(self):
        """
        Split slides into categories based on their functional purpose.
        """
        if pexists(self.split_cache):
            split = json.load(open(self.split_cache))
            return set(split["content_slides_index"]), split["functional_cluster"]
        category_split_template = Template(open("prompts/category_split.txt").read())
        functional_cluster = self.language_model(
            category_split_template.render(slides=self.prs.to_text()),  
            return_json=True,
        )   
        functional_slides = set(sum(functional_cluster.values(), []))
        content_slides_index = set(range(1, len(self.prs) + 1)) - functional_slides

        json.dump(
            {
                "content_slides_index": list(content_slides_index),
                "functional_cluster": functional_cluster,
            },
            open(self.split_cache, "w"),
            indent=4,
            ensure_ascii=False,
        )
        return content_slides_index, functional_cluster

    def layout_split(self, content_slides_index: set[int]):
        """
        Cluster slides into different layouts.
        """
        embeddings = get_image_embedding(self.template_image_folder, *self.image_models)    
        assert len(embeddings) == len(self.prs)
        template = Template(open("prompts/ask_category.txt").read())
        content_split = defaultdict(list)
        for slide_idx in content_slides_index:  
            slide = self.prs.slides[slide_idx - 1]
            content_type = slide.get_content_type()
            layout_name = slide.slide_layout_name
            content_split[(layout_name, content_type)].append(slide_idx)

        for (layout_name, content_type), slides in content_split.items():
            sub_embeddings = [
                embeddings[f"slide_{slide_idx:04d}.jpg"] for slide_idx in slides
            ]
            similarity = images_cosine_similarity(sub_embeddings)
            for cluster in get_cluster(similarity):
                slide_indexs = [slides[i] for i in cluster]
                template_id = max(
                    slide_indexs,
                    key=lambda x: len(self.prs.slides[x - 1].shapes),
                )
                cluster_name = (
                    self.vision_model(
                        template.render(
                            existed_layoutnames=list(self.slide_induction.keys()),
                        ),
                        pjoin(self.ppt_image_folder, f"slide_{template_id:04d}.jpg"),
                    )
                    + ":"
                    + content_type
                )
                self.slide_induction[cluster_name]["template_id"] = template_id
                self.slide_induction[cluster_name]["slides"] = slide_indexs

    @tenacity
    def content_induct(self):
        """
        Perform content schema extraction for the presentation.
        """
        # self.slide_induction = self.layout_induct()

        full_layout_info = {}
        for i, slide in enumerate(self.prs.slides):
            slide_info = {}
            for shape in slide.shapes:
                info = {
                "pptc_description": getattr(shape, "pptc_description", None),
                "pptc_size_info": getattr(shape, "pptc_size_info", None),
                "pptc_space_info": getattr(shape, "pptc_space_info", None),
                "pptc_text_info": getattr(shape, "pptc_text_info", None),
                "shape_idx": getattr(shape, "shape_idx", None)
                    }
                if 'TextBox' in getattr(shape, "pptc_description", None):
                    info["pptc_text_info"] = info["pptc_text_info"].split('\n\n')[-1]
                
                slide_info[f'shape_{info["shape_idx"]}'] = info
            full_layout_info[f'slide_{i}'] = slide_info


        with open("prompts/content_induct_v2.txt", "r", encoding="utf-8") as f:
            content_induct_prompt = f.read()

        temp_info = str(full_layout_info)
        content_induct_prompt = content_induct_prompt.replace("{{slide_info}}", temp_info)

        schema = self.language_model(content_induct_prompt, return_json=True)

        full_info = {}
        content_induct_prompt = Template(open("prompts/content_induct.txt").read())

        content_simp_prompt = Template(open("prompts/content_induct_v2_simp.txt").read())

        for i, slide in enumerate(self.prs.slides):

            schema_ori = self.language_model(
                    content_induct_prompt.render(
                        slide=slide.to_html(
                            element_id=False, paragraph_id=False
                        )   
                    ),
                    return_json=True,
                )
            
            schema_sim = self.language_model(
                    content_simp_prompt.render(
                        con_scheme = schema_ori,
                        details = full_layout_info[f'slide_{i}']
                    ),
                    return_json=True,
                )
            

            concise_layout = {}
            img_count = 0
            text_box_count = 0

            for key, value in schema_sim.items():
                if value.get('type') == 'image':
                    img_count += 1
                    # 提取宽高
                    size_info = value.get('pptc_size_info', '')
                    match = re.search(r'height=(\d+)pt, width=(\d+)pt', size_info)
                    if match:
                        height = int(match.group(1))
                        width = int(match.group(2))
                        ratio = round(height / width, 3) if width != 0 else None
                        concise_layout[f'image_{img_count}'] = {
                            'size': f'height={height}pt, width={width}pt',
                            'height:width_ratio': ratio
                        }
                if value.get('type') == 'text':
                    text_box_count+=1
                    

            concise_layout['image_num'] = img_count
            concise_layout['text_box_num'] = text_box_count


            for k in list(schema_ori.keys()):
                if "data" not in schema_ori[k]:
                    raise ValueError(f"Cannot find `data` in {k}\n{schema_ori[k]}")
                if len(schema_ori[k]["data"]) == 0:
                    print(f"Empty content schema: {schema_ori[k]}")
                    schema_ori.pop(k)
            assert len(schema_ori) > 0, "No content schema_ori generated"

            full_info[f'slide_{i}'] = {'main_theme': schema[f'slide_{i}'], 'concise_layout': concise_layout, 'content_schema': schema_sim, 'template_id': i+1}
            

        self.slide_induction = full_info
            
        json.dump(
            self.slide_induction,
            open(self.induct_cache, "w"),
            indent=4,
            ensure_ascii=False,
        )  # 
        
        return self.slide_induction
