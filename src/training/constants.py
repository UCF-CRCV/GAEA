# Special token indexes used to handle padding and ignored positions in loss calculation
IGNORE_INDEX = -100

# Special tokens used for message formatting
DEFAULT_IM_START_TOKEN = "<|im_start|>"  
DEFAULT_IM_END_TOKEN = "<|im_end|>"      
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"    
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"    
LLAVA_IMAGE_TOKEN = "<image>"            
LLAVA_VIDEO_TOKEN = "<video>"            
VISION_START_TOKEN = "<|vision_start|>"  
VISION_END_TOKEN = "<|vision_end|>"      

# Default system message for conversation initialization
SYSTEM_MESSAGE = "You are a helpful assistant."

def dynamic_prompt(prompt_type):
    """
    Returns specialized instruction prompts based on the type of question.
    
    Args:
        prompt_type (str): Type of prompt to generate (LOC, LVQA, SVQA, MCQ, TF, GEN)
            - LOC: Location identification
            - LVQA: Long-form visual question answering
            - SVQA: Short-form visual question answering
            - MCQ: Multiple choice questions
            - TF: True/False questions
            - GEN: General geography and tourism expertise
    
    Returns:
        str: The specialized prompt for the given question type
    """
    if prompt_type == 'LOC':
        return 'As a geography and tourism expert, analyze the image to determine its exact location. Utilize your extensive knowledge of geography, terrain, landscapes, flora, fauna, infrastructure, and recognizable landmarks to identify the city and country where the image was taken. Question: '
    elif prompt_type == 'LVQA':
        return 'Drawing upon your expertise in geography and tourism, examine the image and provide a comprehensive description of the community or lifestyle depicted. Include insights about cultural practices, geographic features, terrain, local flora and fauna, infrastructure, and any natural or man-made elements that characterize the location. Consider how these factors influence the lifestyle and community in the area. Question: '
    elif prompt_type == 'SVQA':
        return 'Provide a short answer on notable landmarks, museums, parks, restaurants, or activities that visitors might enjoy in the area. Highlight amenities and services that enhance the tourism experience at this location. Question: '
    elif prompt_type == 'MCQ':
        return 'Use your comprehensive knowledge of geography, landmarks, and tourism to analyze the image and determine the correct answer from the options provided. Note, your final answer should be a choice of either A, B, C, or D, including both the letter and the complete text of the option exactly as presented. Question: '
    elif prompt_type == 'TF':
        return "Use your comprehensive knowledge of notable landmarks, museums, parks, restaurants, and related attractions to evaluate the following statement. Provide your final answer as either 'True' or 'False'. Question: "
    elif prompt_type == 'GEN':
        return 'You are an expert in geography and tourism. You possess extensive knowledge of geography, terrain, landscapes, flora, fauna, infrastructure, and other natural or man-made features that help determine a location from images or descriptions. Additionally, you are well-versed in tourism-related information, including amenities such as hotels, restaurants, attractions, and services available in various locations. Question: '
