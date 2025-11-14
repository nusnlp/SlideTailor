import json
import os
import traceback
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime

import jsonlines
import PIL.Image
import torch
from FlagEmbedding import BGEM3FlagModel
from jinja2 import Environment, StrictUndefined
from rich import print

from apis import API_TYPES, CodeExecutor
from llms import Role
from model_utils import get_text_embedding
from presentation import Presentation, SlidePage
from utils import Config, get_slide_content, pexists, pjoin, tenacity


@dataclass
class PPTGen(ABC):
    """
    Stage II: Presentation Generation
    An abstract base class for generating PowerPoint presentations.
    It accepts a reference presentation as input, then generates a presentation outline and slides.
    """

    roles: list[str] = field(default_factory=list)

    def __init__(
        self,
        vision_model,
        language_model,
        text_model: BGEM3FlagModel,
        retry_times: int = 3,
        force_pages: bool = False,
        error_exit: bool = True,
        record_cost: bool = True,
        **kwargs,
    ):
        """
        Initialize the PPTGen.

        Args:
            text_model (BGEM3FlagModel): The text model for generating content.
            retry_times (int): The number of times to retry failed actions.
            force_pages (bool): Whether to force a specific number of pages.
            error_exit (bool): Whether to exit on error.
            record_cost (bool): Whether to record the cost of generation.
            **kwargs: Additional arguments.
        """
        self.text_model = text_model
        self.retry_times = retry_times
        self.force_pages = force_pages
        self.error_exit = error_exit
        
        self.llm = language_model
        self.vision_model = vision_model
        self.language_model = language_model
        
        self._hire_staffs(record_cost, **kwargs)

    def set_reference(
        self,
        presentation: Presentation,
        slide_induction: dict,
        generation_config: Config,
        pref_guidelines: str = "",
    ):
        """
        Set the reference presentation and extracted presentation information.

        Args:
            presentation (Presentation): The presentation object.
            slide_induction (dict): The slide induction data.

        Returns:
            PPTGen: The updated PPTGen object.
        """
        self.presentation = presentation
        self.config = generation_config

        self.slide_induction = slide_induction
        # do not affect the original slide_induction
        slide_induction = slide_induction.copy() 
        
        self.layout_info = slide_induction
        
        self.layout_keys = self.layout_info.keys()
        self.layout_names = list(self.layout_info.keys())
        
        # save and load a copy of the presentation
        self.presentation.save(pjoin(self.config.RUN_DIR, "empty_prs.pptx"), layout_only=False)
        self.empty_prs = Presentation.from_file(pjoin(self.config.RUN_DIR, "empty_prs.pptx"), self.config)
        # self.empty_prs = deepcopy(presentation)
        
        self.pref_guidelines = pref_guidelines
        
        self.slide_induction_concise = self._make_concise_induction(self.slide_induction)
        
        return self

    def generate_presentation(
        self,
        config: Config,
        images: dict[str, str],
        num_slides: int,
        doc_json: dict[str, str],
        presentation_outline: dict[str, str] | None = None,
    ) -> str | None:
        """
        Generate a PowerPoint presentation.

        Args:
            config (Config): The configuration object.
            images (dict[str, str]): A dictionary of image paths and captions.
            num_slides (int): The number of slides to generate.
            doc_json (dict[str, str]): The document JSON data.

        Save:
            final.pptx: The final PowerPoint presentation to the config.RUN_DIR directory.

        Raise:
            ValueError: if failed to generate presentation outline.
        """
        self.config = config
        self.doc_json = doc_json
        meta_data = "\n".join(
            [f"{k}: {v}" for k, v in self.doc_json.get("metadata", {}).items()]
        )
        self.metadata = (
            f"{meta_data}\nPresentation Time: {datetime.now().strftime('%Y-%m-%d')}\n"
        )
        self.image_information = ""
        for k, v in images.items():
            assert pexists(k), f"Image {k} not found"
            size = PIL.Image.open(k).size
            self.image_information += (
                f"Image path: {k}, size: {size[0]}*{size[1]} px\n caption: {v}\n"
            )
        succ_flag = True
        code_executor = CodeExecutor(self.retry_times)
        
        # if presentation outline is given, use it
        if presentation_outline:
            print(f"generate_presentation with given outline: {presentation_outline}")
            self.outline = presentation_outline
        else:
            if self.pref_guidelines:
                # Use the two-stage outline generation process when preference guidelines are provided
                self.outline = self._generate_outline_2_stage_with_guidelines(num_slides)
            else:
                self.outline = self._generate_outline(num_slides)
            
        self.simple_outline = "\n".join(
            [
                f"Slide {slide_idx+1}: {slide_title}"
                for slide_idx, slide_title in enumerate(self.outline)
            ]
        )
        generated_slides = []
        print(f"generating {num_slides} slides, with {len(self.outline)} outlines")
        for slide_data in enumerate(self.outline.items()):
            if self.force_pages and slide_data[0] == num_slides:
                break
            print(f"generating slide {slide_data[0]+1}...")
            slide = self._generate_slide(slide_data, code_executor, self.config.RUN_DIR)
            if slide is not None:
                generated_slides.append(slide)
                continue
            if self.error_exit:
                succ_flag = False
                break
        self._save_history(code_executor)
                
        # cut off empty_prs.slides by num_slides
        self.empty_prs.slides = self.empty_prs.slides[:num_slides]
        
        if succ_flag:
            self.empty_prs.slides = generated_slides
            self.empty_prs.save(pjoin(self.config.RUN_DIR, "final.pptx"), layout_only=False)
        
            return pjoin(self.config.RUN_DIR, "final.pptx"), self.outline
        else:
            return None, None

    def _save_history(self, code_executor: CodeExecutor):
        """
        Save the history of code execution, API calls and agent steps.
        """
        os.makedirs(pjoin(self.config.RUN_DIR, "history"), exist_ok=True)
        for role in self.staffs.values():
            role.save_history(pjoin(self.config.RUN_DIR, "history"))
            role.history = []
        if len(code_executor.code_history) == 0:
            return
        with jsonlines.open(
            pjoin(self.config.RUN_DIR, "code_steps.jsonl"), "w"
        ) as writer:
            writer.write_all(code_executor.code_history)
        with jsonlines.open(
            pjoin(self.config.RUN_DIR, "agent_steps.jsonl"), "w"
        ) as writer:
            writer.write_all(code_executor.api_history)

    @tenacity
    def _generate_outline(self, num_slides: int):
        """
        Generate an outline for the presentation.

        Args:
            num_slides (int): The number of slides to generate.

        Returns:
            dict: The generated outline.
        """
        outline_file = pjoin(self.config.RUN_DIR, "presentation_outline.json")
        doc_overview = deepcopy(self.doc_json)
        for section in doc_overview["sections"]:
            [sub.pop("content") for sub in section["subsections"]]

        outline = self.staffs["planner"](
            num_slides=num_slides,
            functional_keys=str(self.layout_info),
            json_content=doc_overview,
            image_information=self.image_information,
            pref_guidelines="None",
        )
        outline = self._valid_outline(outline, num_slides)  # add num of slide validation
        json.dump(
            outline,
            open(outline_file, "w"),
            ensure_ascii=False,
            indent=4,
        )
        return outline
    
    @tenacity
    def _generate_outline_with_guidelines(self, num_slides: int):
        """
        Generate an outline for the presentation.

        Args:
            num_slides (int): The number of slides to generate.

        Returns:
            dict: The generated outline.
        """
        outline_file = pjoin(self.config.RUN_DIR, "presentation_outline.json")
        doc_overview = deepcopy(self.doc_json)
        
        # make the summarized_doc_content compact
        for section in doc_overview["sections"]:
            [sub.pop("content") for sub in section["subsections"]]
        
        # import pdb; pdb.set_trace()

        outline = self.staffs["planner"](
            num_slides=num_slides,
            functional_keys=str(self.layout_info),
            # functional_keys="\n".join(self.layout_info),
            summarized_doc_content=doc_overview,
            image_information=self.image_information,
            pref_guidelines=self.pref_guidelines,
        )
        outline = self._valid_outline(outline, num_slides)  # add num of slide validation
        json.dump(
            outline,
            open(outline_file, "w"),
            ensure_ascii=False,
            indent=4,
        )
        
        return outline
    
    @tenacity
    def _generate_outline_2_stage_with_guidelines(self, num_slides: int):
        """
        Generate an outline for the presentation using a 2-stage approach:
        1. Stage 1: Generate content outline focusing on content quality and coherence
        2. Stage 2: Refine the layout selections based on the content outline

        Args:
            num_slides (int): The number of slides to generate.

        Returns:
            dict: The final presentation outline with optimized content and layouts.
        """
        
        content_outline_file = pjoin(self.config.RUN_DIR, "presentation_content_outline.json")
        layout_outline_file = pjoin(self.config.RUN_DIR, "presentation_layout_outline.json")
        presentation_outline_file = pjoin(self.config.RUN_DIR, "presentation_outline.json")
        
        doc_overview = deepcopy(self.doc_json)
        
        # Make the summarized_doc_content compact
        for section in doc_overview["sections"]:
            [sub.pop("content") for sub in section["subsections"]]
        
        print("Stage 1: Generating content outline...")
        # Stage 1: Generate content outline using planner_content
        content_outline = self.staffs["planner_content"](
            num_slides=num_slides,
            summarized_doc_content=doc_overview,
            image_information=self.image_information,
            pref_guidelines=self.pref_guidelines,
        )
        
        # Validate content outline
        content_outline = self._valid_content_outline(content_outline, num_slides)
        
        # Save content outline
        json.dump(
            content_outline,
            open(content_outline_file, "w"),
            ensure_ascii=False,
            indent=4,
        )
        
        print("Stage 2: Refining layout selections...")
        # Stage 2: Refine layout selections using planner_layout
        layout_outline = self.staffs["planner_layout"](
            content_outline=content_outline,
            functional_keys=str(self.layout_info),
        )
        
        # Validate layout outline
        layout_outline = self._valid_layout_outline(layout_outline, num_slides)
        
        # Save layout outline
        json.dump(
            layout_outline,
            open(layout_outline_file, "w"),
            ensure_ascii=False,
            indent=4,
        )
        
        final_presentation_outline = layout_outline
        
        json.dump(
            final_presentation_outline,
            open(presentation_outline_file, "w"),
            ensure_ascii=False,
            indent=4,
        )
        
        
        return final_presentation_outline
    
    def _valid_content_outline(self, outline: dict, num_slides: int, retry: int = 0) -> dict:
        """
        Validate the content outline generated in Stage 1.
        """
        len_outline_check = len(outline) == num_slides
        
        if not len_outline_check:
            feedback = f"Number of slides in content outline ({len(outline)}) does not match the expected number ({num_slides})"
            traceback = f"Content outline validation failed: {feedback}"
            print(f"Re-generating content outline: {feedback}")
            
            new_outline = self.staffs["planner_content"].retry(
                feedback,
                traceback,
                retry + 1
            )
            return self._valid_content_outline(new_outline, num_slides, retry + 1)
        
        return outline
    
    def _valid_layout_outline(self, outline: dict, num_slides: int, retry: int = 0) -> dict:
        """
        Validate the layout outline generated in Stage 2.
        
        Args:
            outline (dict): The layout outline to validate.
            num_slides (int): The expected number of slides.
            retry (int): The current retry count.
            
        Returns:
            dict: The validated layout outline.
        """
        used_layouts = [page['layout'] for page in outline.values() if 'layout' in page]
        layout_check = all(layout in self.layout_keys for layout in used_layouts)
        len_outline_check = len(outline) == num_slides
        
        # If either validation fails, generate unified feedback
        if not layout_check or not len_outline_check:
            feedback_parts = []
            if not layout_check:
                invalid_layouts = [layout for layout in used_layouts if layout not in self.layout_keys]
                feedback_parts.append(f"Invalid layouts: {invalid_layouts}. Layouts must be in {self.layout_keys}")
            if not len_outline_check:
                feedback_parts.append(f"Number of slides in layout outline ({len(outline)}) does not match the expected number ({num_slides})")
            
            feedback = " AND ".join(feedback_parts)
            traceback = f"Layout outline validation failed: {feedback}"
            print(f"Re-generating layout outline: {feedback}")
            
            new_outline = self.staffs["planner_layout"].retry(
                feedback,
                traceback,
                retry + 1
            )
            return self._valid_layout_outline(new_outline, num_slides, retry + 1)
        
        return outline

    def _valid_outline(self, outline: dict, num_slides: int, retry: int = 0) -> dict:
        """
        Validate the generated outline for the single-stage process.
        Also validate if num of outline is equal to the number of slides

        Raises:
            ValueError: If the outline is invalid.
        """
        
        used_layouts = [page['layout'] for page in outline.values() if 'layout' in page]
        layout_check = all(layout in self.layout_keys for layout in used_layouts)
        len_outline_check = len(outline) == num_slides
        
        # If either validation fails, generate unified feedback
        if not layout_check or not len_outline_check:
            feedback_parts = []
            if not layout_check:
                feedback_parts.append(f"Layouts ({used_layouts}) must be in {self.layout_keys}")
            if not len_outline_check:
                feedback_parts.append(f"Number of outlines ({len(outline)}) does not match the number of slides ({num_slides})")
            
            feedback = " AND ".join(feedback_parts)
            traceback = f"Outline validation failed: {feedback}"
            print(f"Re-generating outline: {feedback}")
            
            new_outline = self.staffs["planner"].retry(
                feedback,
                traceback,
                retry + 1
            )
            return self._valid_outline(new_outline, num_slides, retry + 1)
        
        return outline

    def _hire_staffs(self, record_cost: bool, **kwargs) -> dict[str, Role]:
        """
        Initialize agent roles and their models
        """
        jinja_env = Environment(undefined=StrictUndefined)
        self.staffs = {
            role: Role(
                role,
                env=jinja_env,
                record_cost=record_cost,
                text_model=self.text_model,
                llm=self.llm,
                **kwargs,
            )
            for role in self.roles
        }


    def _generate_slide(self, slide_data, code_executor: CodeExecutor, run_dir=None) -> SlidePage:
        """
        Generate a slide from the slide data.
        """
        slide_idx, (slide_title, slide) = slide_data
        images_info = "No Images"
        
        # check if the layout is available
        if slide["layout"] not in self.slide_induction.keys():
            print(f"layout {slide['layout']} not found, please check the outline")
            import pdb; pdb.set_trace()
            # todo: regen ouline / validate outline
        
        template = deepcopy(self.slide_induction[slide["layout"]])  
        
        try:
            return self.synergize(
                template,
                slide_data,
                code_executor,
                images_info,
            )
        except Exception as e:
            # import pdb; pdb.set_trace()
            print(f"generate slide {slide_idx} failed: {e}")
            print(traceback.format_exc())
            print(self.config.RUN_DIR)


class PPTCrew(PPTGen):
    """
    A class to generate PowerPoint presentations with a crew of agents.
    """

    roles: list[str] = ["planner", "planner_content", "planner_layout", "editor", "coder"]

    def synergize(
        self,
        template: dict,
        slide_content: str,
        code_executor: CodeExecutor,
        images_info: str,
    ) -> SlidePage:
        """
        Synergize Agents to generate a slide.

        Args:
            template (dict): The template data.
            slide_content (str): The slide content.
            code_executor (CodeExecutor): The code executor object.
            images_info (str): The image information.

        Returns:
            SlidePage: The generated slide.
        """
        content_schema = template["content_schema"]
        old_data = self._prepare_schema(content_schema)
        
        # import pdb; pdb.set_trace()
        
        editor_output = self.staffs["editor"](
            schema=content_schema,
            simple_outline=self.simple_outline,    
            metadata=self.metadata,     # 
            text=slide_content,     
            # images_info=images_info,
        )
        
        print(f"editor_output (editor): {editor_output}")
        
        command_list = self._generate_commands(
            editor_output,
            content_schema,
            old_data  
        ) 

        edit_actions = self.staffs["coder"](
            api_docs=code_executor.get_apis_docs(API_TYPES.Agent.value),
            edit_target=self.presentation.slides[template["template_id"] - 1].to_html(),
            command_list="\n".join([str(i) for i in command_list]),
        )
        
        print(f"edit_actions (coder): {edit_actions}")
        
        for error_idx in range(self.retry_times):
            edited_slide: SlidePage = deepcopy(
                self.presentation.slides[template["template_id"] - 1]
            )
            feedback = code_executor.execute_actions(edit_actions, edited_slide)
            if feedback is None:
                break
            if error_idx == self.retry_times:
                raise Exception(
                    f"Failed to generate slide, tried too many times at editing\ntraceback: {feedback[1]}"
                )
            edit_actions = self.staffs["coder"].retry(*feedback, error_idx + 1)
        self.empty_prs.build_slide(edited_slide)
        return edited_slide

    def _prepare_schema(self, content_schema: dict):
        """
        Prepare the content schema for editing.

        Args:
            content_schema (dict): The content schema.

        Returns:
            dict: The old data extracted from the schema.
        """
        old_data = {}
        if isinstance(content_schema, list):
            if len(content_schema) == 0:
                print(f"content_schema is empty, {content_schema}, \
                    maybe check the outline if it uses available layouts")
                import pdb; pdb.set_trace()
                
            content_schema = {f"element_{i}": el for i, el in enumerate(content_schema)}
            
        for el_name, el_info in content_schema.items():
            if el_info["type"] == "text":
                if not isinstance(el_info["data"], list):
                    el_info["data"] = [el_info["data"]]
                if len(el_info["data"]) > 1:
                    charater_counts = [len(i) for i in el_info["data"]]
                    content_schema[el_name]["suggestedCharacters"] = (
                        str(min(charater_counts)) + "-" + str(max(charater_counts))
                    )
                else:
                    content_schema[el_name]["suggestedCharacters"] = "<" + str(
                        len(el_info["data"][0])
                    )
            old_data[el_name] = el_info.pop("data")
            content_schema[el_name]["default_quantity"] = 1
            if isinstance(old_data[el_name], list):
                content_schema[el_name]["default_quantity"] = len(old_data[el_name])
        assert len(old_data) > 0, "No old data generated"
        return old_data

    def _generate_commands(
        self, editor_output: dict, content_schema: dict, old_data: dict, retry: int = 0
    ):
        """
        Generate commands for editing the slide content.

        Args:
            editor_output (dict): The editor output.
            content_schema (dict): The content schema.
            old_data (dict): The old data.
            retry (int): The number of retries.

        Returns:
            list: A list of commands.

        Raises:
            Exception: If command generation fails.
        """
        command_list = []
        try:
            for el_name, el_data in editor_output.items():
                assert (
                    "data" in el_data
                ), """key `data` not found in output
                        please give your output as a dict like
                        {
                            "element1": {
                                "data": ["text1", "text2"] for text elements
                                or ["/path/to/image", "..."] for image elements
                            },
                        }"""
                charater_counts = [len(i) for i in el_data["data"]]
                max_charater_count = max([len(i) for i in old_data[el_name]])
                if max(charater_counts) > max_charater_count * 1.5:
                    raise ValueError(
                        f"Content for '{el_name}' exceeds character limit ({max(charater_counts)} > {max_charater_count}). "
                        f"Please reduce the content length to maintain slide readability and visual balance. "
                        f"Current text: '{el_data['data']}'"
                    )
        except Exception as e:
            if retry < self.retry_times:
                new_output = self.staffs["editor"].retry(
                    e,
                    traceback.format_exc(),
                    retry + 1,
                )
                return self._generate_commands(
                    new_output, content_schema, old_data, retry + 1
                )

        for el_name, old_content in old_data.items():
            if not isinstance(old_content, list):
                old_content = [old_content]

            new_content = editor_output.get(el_name, {}).get("data", None)
            if not isinstance(new_content, list):
                new_content = [new_content]

            new_content = [i for i in new_content if i]

            if content_schema[el_name]["type"] == "image":
                new_content = [i for i in new_content if pexists(i)]

            quantity_change = len(new_content) - len(old_content)
            command_list.append(
                (
                    el_name,
                    content_schema[el_name]["type"],
                    f"quantity_change: {quantity_change}",
                    old_content,
                    new_content,
                )
            )

        assert len(command_list) > 0, "No commands generated"
        return command_list
    
    def _make_concise_induction(self, slide_induction):
        """
        only keeps the slide keys and their `main_theme` and `concise_layout` fields.
        """
        concise_dict = {}
        for slide_key, slide_data in slide_induction.items():
            concise_dict[slide_key] = {
                'main_theme': slide_data['main_theme'],
                'concise_layout': slide_data['concise_layout']
            }
        return concise_dict
