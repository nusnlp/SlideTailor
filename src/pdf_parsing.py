import json
import os
import PIL.Image

from pdf2image import convert_from_path
import re

# put into functions
# from marker.config.parser import ConfigParser
# from marker.converters.pdf import PdfConverter
# from marker.output import text_from_rendered
# from marker.schema import BlockTypes

from utils import is_image_path, pjoin
from doc_handling import refine_document


A4_PAGE_WIDTH = 596
A4_PAGE_HEIGHT = 842

CAPTION_PROMPT_PATH = "prompts/caption.txt"

def parse_pdf(
    pdf_path: str,
    output_path: str,
    marker_model: dict,
) -> str:
    """
    Parse a PDF file and extract text and images.
    Returns:
        str: The full text extracted from the PDF.
    """
    full_text, _, _ = parse_with_converter(pdf_path, output_path, marker_model)
    return full_text

def parsing_pdf_with_caption(
    pdf_path: str,
    parsed_pdf_dir: str,
    marker_model: dict,
    vision_model,
    language_model,
    caption=True,
    use_cache=True,
) -> str:
    """
    Parse a PDF file and extract text and images with captions.
    """
    os.makedirs(parsed_pdf_dir, exist_ok=True)
    
    # entrance for no caption
    if not caption:
        return parse_pdf(pdf_path, parsed_pdf_dir, marker_model)
    
    caption_json_path = pjoin(parsed_pdf_dir, "caption.json")
    text_content_path = pjoin(parsed_pdf_dir, "source.md")
    text_content_w_captions_path = pjoin(parsed_pdf_dir, "source_with_captions.md")
    refined_doc_json_path = pjoin(parsed_pdf_dir, "refined_doc.json")


    if not os.path.exists(refined_doc_json_path) or not os.path.exists(text_content_path) or not use_cache:
        print("[INFO] Parsing PDF ...")
        # Get text content and document object in one call
        text_content, document, _ = parse_with_converter(pdf_path, parsed_pdf_dir, marker_model)

        # Extract tables as images using TableImageConverter, reusing the document
        print("[INFO] Extracting tables & equations as images...")
        table_converter = TableImageConverter(artifact_dict=marker_model)
        equation_converter = EquationImageConverter(artifact_dict=marker_model)

        table_results = table_converter(pdf_path, output_dir=parsed_pdf_dir, dpi=150, padding_px=5, document=document)
        equation_results = equation_converter(pdf_path, output_dir=parsed_pdf_dir, dpi=150, padding_px=10, document=document)
        
        if not os.path.exists(caption_json_path):
            with open(CAPTION_PROMPT_PATH, "r", encoding="utf-8") as f:
                caption_prompt = f.read()
            images_info = {}
            
            # Process all image files in the directory (including regular images, tables, and equations)
            for k in os.listdir(parsed_pdf_dir):
                if is_image_path(k):
                    img_path = pjoin(parsed_pdf_dir, k)
                    try:
                        # Use different prompt prefixes based on image type
                        if "table_" in k:
                            specialized_prompt = "This is a table extracted from a document. Briefly describe the content, structure, and purpose of this table. " + caption_prompt
                            text_cap = vision_model(specialized_prompt, [img_path])
                        elif "equation_" in k:
                            specialized_prompt = "This is a mathematical equation extracted from a document. Briefly describe the meaning, components, and purpose of this equation. " + caption_prompt
                            text_cap = vision_model(specialized_prompt, [img_path])
                        else:  # regular images from the pdf
                            text_cap = vision_model(caption_prompt, [img_path])
                            
                        with PIL.Image.open(img_path) as img:
                            size = img.size
                        images_info[img_path] = [text_cap, size]
                    except Exception as e:
                        print(f"[ERROR] Could not caption {k}: {str(e)}")
            
            with open(caption_json_path, "w", encoding="utf-8") as f:
                json.dump(images_info, f, ensure_ascii=False, indent=4)

        doc_json = refine_document(language_model, text_content)
        json.dump(doc_json, open(refined_doc_json_path, "w"), indent=4)
        
        # Add captions to markdown content
        print("[INFO] Adding captions to markdown content...")
        text_content_w_captions = add_captions_to_markdown(text_content, caption_json_path)
        with open(text_content_w_captions_path, "w", encoding="utf-8") as f:
            f.write(text_content_w_captions)
    else:
        print("[INFO] Using cached refined_doc.json")
        doc_json = json.load(open(refined_doc_json_path, "r"))
        text_content = open(text_content_path, "r", encoding="utf-8").read()
        
        # Add captions to markdown content if it doesn't exist
        if not os.path.exists(text_content_w_captions_path):
            print("[INFO] Adding captions to markdown content...")
            text_content_w_captions = add_captions_to_markdown(text_content, caption_json_path)
            with open(text_content_w_captions_path, "w", encoding="utf-8") as f:
                f.write(text_content_w_captions)
        else:
            text_content_w_captions = open(text_content_w_captions_path, "r", encoding="utf-8").read()
    
    return text_content_w_captions


def add_captions_to_markdown(text_content, caption_json_path):
    """
    Enhances the markdown content by replacing image paths with their captions
    
    Args:
        text_content: The original markdown content
        caption_json_path: Path to the caption.json file
        
    Returns:
        Enhanced markdown with image captions
    """
    if not os.path.exists(caption_json_path):
        print("[WARNING] No caption.json found, returning original content")
        return text_content
    
    try:
        with open(caption_json_path, 'r', encoding='utf-8') as f:
            captions_data = json.load(f)
        
        # Create mapping from base filename to caption
        caption_map = {}
        for full_path, caption_info in captions_data.items():
            base_filename = os.path.basename(full_path)
            caption_text = caption_info[0]
            caption_map[base_filename] = caption_text
        
        # Regular expression to find markdown image references
        # Format: ![](<image_filename>)
        image_pattern = r'!\[\]\(([^)]+)\)'
        
        def replace_with_caption(match):
            image_path = match.group(1)
            base_filename = os.path.basename(image_path)
            
            if base_filename in caption_map:
                caption = caption_map[base_filename]
                # Preserve the original image but add a caption
                return f"![{caption}]({image_path})"
            else:
                # Return the original image reference if no caption is found
                return match.group(0)
        
        # Replace all image references with captioned versions
        text_content_w_captions = re.sub(image_pattern, replace_with_caption, text_content)
        
        return text_content_w_captions
    
    except Exception as e:
        print(f"[ERROR] Failed to add captions to markdown: {str(e)}")
        return text_content
    

def parse_with_converter(
    pdf_path: str,
    output_path: str,
    model_lst: list,
) -> tuple:
    """
    Parse a PDF file and extract text, images, and document object.
    This function centralizes PDF conversion to be reused by other functions.

    Args:
        pdf_path (str): The path to the PDF file.
        output_path (str): The directory to save the extracted content.
        model_lst (list): A list of models for processing the PDF.

    Returns:
        tuple: (full_text, document, rendered) where:
            - full_text is the extracted text content
            - document is the document object for further processing
            - rendered is the rendered output
    """
    from marker.config.parser import ConfigParser
    from marker.output import text_from_rendered
    
    os.makedirs(output_path, exist_ok=True)
    config_parser = ConfigParser(
        {
            "output_format": "markdown",
            "workers": 1,
        }
    )
    converter = PdfConverterWrapper(
        config=config_parser.generate_config_dict(),
        artifact_dict=model_lst,
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
    )
    rendered, document = converter(pdf_path)
    full_text, _, images = text_from_rendered(rendered)
    
    # Save text content
    with open(pjoin(output_path, "source.md"), "w+", encoding="utf-8") as f:
        f.write(full_text)
    
    # Save images
    for filename, image in images.items():
        image_filepath = os.path.join(output_path, filename)
        image.save(image_filepath, "JPEG")
    
    # Save metadata
    with open(pjoin(output_path, "meta.json"), "w+") as f:
        f.write(json.dumps(rendered.metadata, indent=4))

    return full_text, document, rendered


class TableImageConverter:
    """
    Table Image Converter finds tables in a PDF document and extracts them as images.
    It uses a combination of methods to identify table regions including:
    1. Analyzing TableConverter output for table cells and rows
    2. Looking for table blocks in the document structure
    3. Falling back to full page extraction if no tables are found
    
    Each table is extracted as a focused image containing just the table, not the entire page.
    """


    def __init__(self, artifact_dict=None, config=None):
        """Initialize the converter with artifacts and configuration"""

        self.artifact_dict = artifact_dict
        self.config = config
        self.tableconverter = None
        
    def find_tables_from_full_document(self, filepath, document=None):
        """
        Process the entire document to identify table regions using multiple methods.
        
        Args:
            filepath (str): Path to the PDF file
            document: Optional pre-processed document object. If provided, it will be used
                    instead of creating a new one.
        """
        from marker.schema import BlockTypes
        
        table_regions = []

        try:
            # Use provided document or create a new one
            if document is None:
                # Create PDF converter to process all document elements
                pdf_converter = PdfConverterWrapper(artifact_dict=self.artifact_dict)
                _, document = pdf_converter(filepath)
            
            for page_idx, page in enumerate(document.pages):
                # Get page size for normalization
                page_width = page.polygon.width
                page_height = page.polygon.height

                # Look for table-like structures in the document
                for child in page.children:
                    if child.block_type == BlockTypes.Table:
                        
                        table = child.model_dump()
                        
                        table_bbox = table['polygon']['bbox']
                        table_page_idx = table['page_id']
                        
                        # Normalize the bounding box by the page size
                        # This makes it portable across different page size representations
                        normalized_bbox = [
                            table_bbox[0] / page_width,
                            table_bbox[1] / page_height,
                            table_bbox[2] / page_width,
                            table_bbox[3] / page_height
                        ]
                        
                        table_regions.append({
                            'page_idx': table_page_idx,
                            'bbox': normalized_bbox, # Normalized by page size
                            'original_bbox': table_bbox, # Original bbox in page coordinates
                            'block_id': table['block_id'],
                            'source': 'table_converter',
                            'page_size': (page_width, page_height) # Store page size for reference
                        })
                                            
        except Exception as e:
            print(f"Error extracting tables from document: {str(e)}")
            # import pdb; pdb.set_trace()
            return []
        
        print(f"Found {len(table_regions)} table regions")
        
        return table_regions
        
    def extract_table_images(self, pdf_path, table_regions, output_dir=None, dpi=100, padding_px=5):
        """
        Extract images for each table region from the PDF
        
        Args:
            pdf_path: Path to the PDF file
            table_regions: List of table regions with page_idx, bbox, and block_id
            output_dir: Directory to save the extracted images
            dpi: DPI for PDF rendering
            
        Returns:
            Dictionary mapping filenames to PIL Image objects
        """
        if not table_regions:
            print("No table regions found with bounding boxes.")
            return {}
        
        # Convert PDF pages to images
        pdf_images = convert_from_path(pdf_path, dpi=dpi)
        
        table_images = {}
        for idx, region in enumerate(table_regions):
            page_idx = region['page_idx']
            bbox = region['bbox']  # This is now normalized
            block_id = region['block_id']
            source = region.get('source', 'unknown')
            
            if page_idx >= len(pdf_images):
                print(f"Skipping table on page {page_idx+1} - page out of range")
                continue
                
            # Get page image
            page_img = pdf_images[page_idx]
            width, height = page_img.size
            
            x0, y0, x1, y1 = bbox
            
            # Skip too small table regions (h or w is less than 5% of the page)
            if x1 - x0 < 0.05 or y1 - y0 < 0.05:
                print(f"Skipping too small table with region: {x0},{y0},{x1},{y1}")
                continue
            
            # Convert normalized coordinates to pixel coordinates directly
            x0 = max(0, int(bbox[0] * width) - padding_px)
            y0 = max(0, int(bbox[1] * height) - padding_px)
            x1 = min(width, int(bbox[2] * width) + padding_px)
            y1 = min(height, int(bbox[3] * height) + padding_px)
            
            # Skip invalid regions
            if x0 >= x1 or y0 >= y1:
                print(f"Skipping invalid table with region: {x0},{y0},{x1},{y1}")
                continue
                

                
            # Crop the table region
            try:
                table_img = page_img.crop((x0, y0, x1, y1))
                
                # Validate this looks like a table - check for minimum size and content
                img_width, img_height = table_img.size
                
                # Generate a filename for the table image indicating source
                img_filename = f"_page_{page_idx+1}_Table_{idx+1}.png"
                
                # Save the image if output_dir is provided
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    img_path = os.path.join(output_dir, img_filename)
                    table_img.save(img_path)
                
                # Add to the dictionary
                table_images[img_filename] = table_img
                print(f"Extracted table {idx+1} from page {page_idx+1} [{source}]: {x0},{y0} to {x1},{y1}")
            except Exception as e:
                print(f"Error extracting table region: {str(e)}")
        
        return table_images
                
    def __call__(self, filepath, output_dir=None, format_ext='json', dpi=150, padding_px=5, document=None):
        """
        Process a PDF document to extract tables and their images
        
        Args:
            filepath: Path to the PDF file
            output_dir: Directory to save the extracted tables and images
            format_ext: Output format for table data ('json' or other format)
            dpi: DPI for PDF rendering
            padding_px: Padding in pixels to add around the table regions
            document: Optional pre-processed document object
            
        Returns:
            Dictionary containing:
                - tables: Extracted table data
                - images: Dictionary mapping filenames to PIL Image objects
        """
        try:
                        
            # Extract table regions if any tables were found
            table_regions = self.find_tables_from_full_document(filepath, document)
            
            # Extract table images if table regions were found
            image_dict = {}
            if table_regions:
                image_dict = self.extract_table_images(filepath, table_regions, output_dir, dpi, padding_px)
                print(f"Extracted {len(image_dict)} table images")
                return {
                    'tables': table_regions,
                    'images': image_dict
                }
            else:
                print("No table regions found for image extraction")
                return {
                    'tables': [],
                    'images': {}
                }
                
            
        except Exception as e:
            print(f"Error processing document: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'tables': [],
                'images': {}
            }

class EquationImageConverter:
    """
    Equation Image Converter finds equations in a PDF document and extracts them as images.
    It uses document structure analysis to identify equation regions including:
    1. Looking for equation blocks in the document structure
    2. Extracting these as focused images containing just the equation, not the entire page.
    
    Each equation is extracted as a separate image that can be used for further processing
    or display in other documents.
    """

    def __init__(self, artifact_dict=None, config=None):
        """Initialize the converter with artifacts and configuration"""
        self.artifact_dict = artifact_dict
        self.config = config
        
    def find_equations_from_document(self, filepath, document=None):
        """
        Process the document to identify equation regions.
        
        Args:
            filepath (str): Path to the PDF file
            document: Optional pre-processed document object. If provided, it will be used
                    instead of creating a new one.
        """
        equation_regions = []

        try:
            from marker.schema import BlockTypes
            
            # Use provided document or create a new one
            if document is None:
                # Create PDF converter wrapper to process all document elements
                pdf_converter = PdfConverterWrapper(artifact_dict=self.artifact_dict)
                _, document = pdf_converter(filepath)
            
            for page_idx, page in enumerate(document.pages):
                # Get page size for normalization
                page_width = page.polygon.width
                page_height = page.polygon.height

                # Look for equation structures in the document
                for child in page.children:
                    if child.block_type == BlockTypes.Equation:
                        
                        equation = child.model_dump()
                        
                        equation_bbox = equation['polygon']['bbox']
                        equation_page_idx = equation['page_id']
                        
                        # Normalize the bounding box by the page size
                        normalized_bbox = [
                            equation_bbox[0] / page_width,
                            equation_bbox[1] / page_height,
                            equation_bbox[2] / page_width,
                            equation_bbox[3] / page_height
                        ]
                        
                        equation_regions.append({
                            'page_idx': equation_page_idx,
                            'bbox': normalized_bbox, # Normalized by page size
                            'original_bbox': equation_bbox, # Original bbox in page coordinates
                            'block_id': equation['block_id'],
                            'source': 'equation_converter',
                            'page_size': (page_width, page_height), # Store page size for reference
                            'latex': equation.get('html', '') # Get LaTeX if available
                        })
                                            
        except Exception as e:
            print(f"Error extracting equations from document: {str(e)}")
            return []
        
        print(f"Found {len(equation_regions)} equation regions")
        
        return equation_regions
        
    def extract_equation_images(self, pdf_path, equation_regions, output_dir=None, dpi=150, padding_px=10, save_latex=False):
        """
        Extract images for each equation region from the PDF
        
        Args:
            pdf_path: Path to the PDF file
            equation_regions: List of equation regions with page_idx, bbox, and block_id
            output_dir: Directory to save the extracted images
            dpi: DPI for PDF rendering
            padding_px: Padding in pixels to add around the equation region
            
        Returns:
            Dictionary mapping filenames to PIL Image objects
        """
        if not equation_regions:
            print("No equation regions found with bounding boxes.")
            return {}
        
        # Convert PDF pages to images
        from pdf2image import convert_from_path
        pdf_images = convert_from_path(pdf_path, dpi=dpi)
        
        equation_images = {}
        for idx, region in enumerate(equation_regions):
            page_idx = region['page_idx']
            bbox = region['bbox']  # This is normalized
            block_id = region['block_id']
            source = region.get('source', 'unknown')
            latex = region.get('latex', '')
            
            if page_idx >= len(pdf_images):
                print(f"Skipping equation on page {page_idx+1} - page out of range")
                continue
                
            # Get page image
            page_img = pdf_images[page_idx]
            width, height = page_img.size
            
            x0, y0, x1, y1 = bbox
            
            # Skip too small equation regions (h or w is less than 1% of the page)
            # Equations are often smaller than tables, so we use a smaller threshold
            if x1 - x0 < 0.01 or y1 - y0 < 0.01:
                print(f"Skipping too small equation with region: {x0},{y0},{x1},{y1}")
                continue
            
            # Convert normalized coordinates to pixel coordinates directly
            x0 = max(0, int(bbox[0] * width) - padding_px)
            y0 = max(0, int(bbox[1] * height) - padding_px)
            x1 = min(width, int(bbox[2] * width) + padding_px)
            y1 = min(height, int(bbox[3] * height) + padding_px)
            
            # Skip invalid regions
            if x0 >= x1 or y0 >= y1:
                print(f"Skipping invalid equation with region: {x0},{y0},{x1},{y1}")
                continue
                
            # Crop the equation region
            try:
                equation_img = page_img.crop((x0, y0, x1, y1))
                
                # Generate a filename for the equation image
                img_filename = f"_page_{page_idx+1}_Equation_{idx+1}.png"
                
                # Save the image if output_dir is provided
                if output_dir:
                    import os
                    os.makedirs(output_dir, exist_ok=True)
                    img_path = os.path.join(output_dir, img_filename)
                    equation_img.save(img_path)
                    
                    # Also save the LaTeX if available
                    if save_latex and latex:
                        latex_filename = os.path.join(output_dir, f"_page_{page_idx+1}_Equation_{idx+1}.tex")
                        with open(latex_filename, 'w', encoding='utf-8') as f:
                            f.write(latex)
                
                # Add to the dictionary
                equation_images[img_filename] = equation_img
                print(f"Extracted equation {idx+1} from page {page_idx+1} [{source}]: {x0},{y0} to {x1},{y1}")
            except Exception as e:
                print(f"Error extracting equation region: {str(e)}")
        
        return equation_images
                
    def __call__(self, filepath, output_dir=None, dpi=150, padding_px=10, document=None):
        """
        Process a PDF document to extract equations and their images
        
        Args:
            filepath: Path to the PDF file
            output_dir: Directory to save the extracted equations and images
            dpi: DPI for PDF rendering
            padding_px: Padding in pixels to add around the equation region
            document: Optional pre-processed document object
            
        Returns:
            Dictionary containing:
                - equations: Extracted equation data
                - images: Dictionary mapping filenames to PIL Image objects
        """
        try:
            # Extract equation regions
            equation_regions = self.find_equations_from_document(filepath, document)
            
            # Extract equation images if equation regions were found
            image_dict = {}
            if equation_regions:
                image_dict = self.extract_equation_images(filepath, equation_regions, output_dir, dpi, padding_px)
                print(f"Extracted {len(image_dict)} equation images")
                return {
                    'equations': equation_regions,
                    'images': image_dict
                }
            else:
                print("No equation regions found for image extraction")
                return {
                    'equations': [],
                    'images': {}
                }
                
        except Exception as e:
            print(f"Error processing document: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'equations': [],
                'images': {}
            }

class PdfConverterWrapper:
    """
    A wrapper class for PdfConverter that returns both renderer and document objects.
    This allows accessing the document object directly without modifying the original
    PdfConverter class in the package.
    """
    
    def __init__(self, artifact_dict=None, config=None, processor_list=None, renderer=None):
        """Initialize the wrapper with the same parameters as PdfConverter"""
        from marker.converters.pdf import PdfConverter
        from marker.providers.pdf import PdfProvider
        from marker.builders.layout import LayoutBuilder
        from marker.builders.line import LineBuilder
        from marker.builders.ocr import OcrBuilder
        from marker.builders.document import DocumentBuilder
        from marker.builders.structure import StructureBuilder
        
        self.artifact_dict = artifact_dict
        self.config = config
        self.processor_list = processor_list
        self.renderer = renderer
        
        # Store class references
        self.PdfProvider = PdfProvider
        self.LayoutBuilder = LayoutBuilder
        self.LineBuilder = LineBuilder
        self.OcrBuilder = OcrBuilder
        self.DocumentBuilder = DocumentBuilder
        self.StructureBuilder = StructureBuilder
        
        # Create the underlying PdfConverter instance
        self.pdf_converter = PdfConverter(
            artifact_dict=self.artifact_dict,
            config=self.config,
            processor_list=self.processor_list,
            renderer=self.renderer
        )
    
    def __call__(self, filepath):
        """
        Process the PDF file and return both renderer and document objects.
        
        Args:
            filepath (str): Path to the PDF file
            
        Returns:
            tuple: (renderer, document) where renderer is the output of the original
                    PdfConverter and document is the Document object
        """
        # Directly implement the same steps as in PdfConverter.__call__
        # without trying to access private methods
        
        # Create PDF provider
        pdf_provider = self.PdfProvider(filepath, self.config)
        
        # Create and resolve layout and OCR builders
        layout_builder = self.pdf_converter.resolve_dependencies(self.LayoutBuilder)
        line_builder = self.pdf_converter.resolve_dependencies(self.LineBuilder)
        ocr_builder = self.pdf_converter.resolve_dependencies(self.OcrBuilder)
        
        # Build document
        document = self.DocumentBuilder(self.config)(pdf_provider, layout_builder, line_builder, ocr_builder)
        
        structure_builder_cls = self.pdf_converter.resolve_dependencies(self.StructureBuilder)
        
        # Apply structure builder
        structure_builder_cls(document)
        
        # Apply processors
        for processor_cls in self.pdf_converter.processor_list:
            processor_cls(document)
        
        # Render document
        renderer = self.pdf_converter.resolve_dependencies(self.pdf_converter.renderer)
        rendered = renderer(document)
        
        # Return both the rendered output and the document
        return rendered, document

