from genai_at_work import config, logutil
logger = logutil.logging.getLogger(__name__)

import traceback
import time 
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import os

HF_CACHE_DIR=config.config["HF_CACHE_DIR"] #"resources/models"
print("hf_cache_dir: ", HF_CACHE_DIR)

def distil_whisper(filepath:str,chatbot:list=[],history:list=[], 
                     model_id:str= "distil-whisper/distil-large-v2",
                     device:str=None,
                     temperature:float=0.2,
                     do_sample:bool=True,
                     low_cpu_mem_usage:bool=False):

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    t0=time.perf_counter()
    cpu_count = int(os.cpu_count()/2)
    result=[]    
    try:
        ## use best device GPU first    
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, 
            low_cpu_mem_usage=low_cpu_mem_usage, 
            use_safetensors=True,
            temperature=temperature,
            do_sample=do_sample,
            cache_dir=config.config["HF_CACHE_DIR"]
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=config.config["HF_CACHE_DIR"])
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            torch_dtype=torch_dtype,
            device=device
        )    
        result = pipe(filepath)
    except RuntimeError as re:
        print ("runtime error switch to CPU", re.args)
        ## fallback use CPU
        if "CUDA failed with error out of memory" in str(re.args):
            device="cpu"
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, torch_dtype=torch.float32, 
                low_cpu_mem_usage=False, 
                use_safetensors=True,
                temperature=temperature,
                do_sample=do_sample ,
                cache_dir=config.config["HF_CACHE_DIR"]               
            )
            model.to(device)
            processor = AutoProcessor.from_pretrained(model_id)
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                max_new_tokens=128,
                torch_dtype=torch_dtype,
                device=device,
                cache_dir=config.config["HF_CACHE_DIR"]
            )    
            result = pipe(filepath)
    # update chatbot
    result_text = result["text"]            
    if chatbot is None:
        chatbot=[]
    chatbot.append((f"trascribed file {filepath}", result_text))
    # update history
    if history is None:
        history=[]
    history.append({"role": "user", "content": f"trascribed file {filepath}\n{result_text}"})                
    t1=time.perf_counter()            
    took=(t1-t0)    
    print(f"distil_whisper took {took}s")    
    return chatbot, history

def faster_whisper(filepath:str,chatbot:list=[],history:list=[], 
                     model_id:str= "large-v3",
                     task: str = "transcribe",
                     device:str="auto", 
                     compute_type:str="default",                     
                     timeline:bool=False, 
                     beam_size:int=6,
                     download_root:str=config.config["HF_CACHE_DIR"],
                     local_files_only:bool=False):
    
    from faster_whisper import WhisperModel
    import os
    import time 
    t0=time.perf_counter()
    ## Run on GPU device="cuda", compute_type="float16"
    cpu_count = int(os.cpu_count()/2)
    result=[]    
    try:
        model = WhisperModel(model_size_or_path=model_id, device=device, cpu_threads=cpu_count, download_root=download_root)
        segments, info = model.transcribe(filepath, task = task, beam_size=beam_size,vad_filter=True)        
        logger.info(f"Detected language {info.language} with probability {info.language_probability:.2f}")
        for segment in segments:
            if timeline:
                result.append(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
            else:
                result.append(str(segment.text)+" ")
    except RuntimeError as re:
        print ("runtime error switch to CPU", re.args)
        print(traceback.format_exc())
        logger.error(f"error occurred {re.args} at {str(traceback.format_exc())}")        
        if "CUDA failed with error out of memory" in str(re.args):
            model = WhisperModel(model_size_or_path=model_id, device="cpu", cpu_threads=cpu_count,download_root=download_root)
            segments, info = model.transcribe(filepath, task = task, beam_size=beam_size,vad_filter=True)
            for segment in segments:
                if timeline:
                    result.append(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
                else:
                    result.append(str(segment.text)+" ")
    t1=time.perf_counter()            
    took=(t1-t0)    
    ## update chatbot
    result_text = " ".join(result).strip()
    if chatbot is None:
        chatbot=[]    
    chatbot.append((f"trascribed file {filepath}", result_text))
    ## update history
    if history is None:
        history=[]    
    history.append({"role": "user", "content": f"trascribed file {filepath}\n{result_text}"})            
    print(f"faster_whisper took {took}s")
    return chatbot, history

def transcribe_video(filepath:str,chatbot:list=[],history:list=[],
                     video_format:str="mp4",
                     export_audio:str="video_audio.wav", 
                     export_format:str="wav", 
                     export_transcriber:str="distil_whisper"):
    from pydub import AudioSegment
    import os
    ## load the video file
    video = AudioSegment.from_file(filepath, format=video_format)
    audio = video.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    audio.export(export_audio, format=export_format)
    ## export
    if export_transcriber=="distil_whisper":
       return distil_whisper(filepath,chatbot,history) 
    elif export_transcriber=="faster_whisper":
       return faster_whisper(filepath,chatbot,history) 
    else:
        raise ValueError(f"unsupported audio export transcriber {export_transcriber}")

def youtube_download(url:str,local_storage:str="./workspace/downloads"):
    from pytube import YouTube    
    import time    
    if not os.path.exists(local_storage):
        os.mkdir(local_storage)
    t0 = time.perf_counter()     
    local_file = None   
    try:
        link = YouTube(url)
        local_file = link.streams.filter(only_audio=True)[0].download(local_storage)
    except Exception as e:
        print(f"youtube_download error:",e)    
        print(traceback.format_exc())        
        logger.error(f"error occurred {e.args} at {str(traceback.format_exc())}")                
        raise e 
    took_in_seconds = time.perf_counter() - t0  
    status_msg=f"youtube_download [{url}] took {took_in_seconds:.2f} seconds"   
    print(status_msg) 
    return local_file       

def transcribe_youtube(input_url:str, chatbot:list=[],history:list=[],
                       export_transcriber:str="distil_whisper"):
    import os
    import time
    t0 = time.perf_counter()    
    local_filepath=None        
    try:
        local_filepath = youtube_download(input_url)
        if export_transcriber=="distil_whisper":
            chatbot, history = distil_whisper(local_filepath,chatbot,history) 
        elif export_transcriber=="faster_whisper":
            chatbot, history = faster_whisper(local_filepath,chatbot,history) 
        else:
            raise ValueError(f"unsupported audio transcriber {export_transcriber}")
        took_in_seconds = time.perf_counter()-t0        
        status_msg=f"transcribe_youtube finished in {took_in_seconds:.2f} seconds"   
    except Exception as e:
        print("transcribe exception ", e)
        print(traceback.format_exc())        
        logger.error(f"error occurred {e.args} at {str(traceback.format_exc())}")                        
    finally:
        ...  # skip clean up to save download time 
        # if local_filepath is not None:
        #     os.remove(local_filepath) 
    return chatbot, history     

def download_youtube_transcript(video_id:str,languages:dict=['en'], mode:str="auto"):
    from youtube_transcript_api import YouTubeTranscriptApi
    # the base class to inherit from when creating your own formatter.
    from youtube_transcript_api.formatters import Formatter
    # some provided subclasses, each outputs a different string format.
    from youtube_transcript_api.formatters import JSONFormatter
    from youtube_transcript_api.formatters import TextFormatter
    from youtube_transcript_api.formatters import WebVTTFormatter
    from youtube_transcript_api.formatters import SRTFormatter

    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    transcript=""    
    # filter for manually created transcripts
    if mode=="manual":
        transcript = transcript_list.find_manually_created_transcript(languages)
    else:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)        
        # or automatically generated ones
        # transcript = transcript_list.find_generated_transcript(languages)
    text_formatter = TextFormatter()
    result_text = text_formatter.format_transcript(transcript)
    return result_text

def download_youtube_translate(video_id:str,languages:dict=['en'], translate_lang:str=None):
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api.formatters import TextFormatter    
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)    
    transcript = transcript_list.find_transcript(languages)
    ##print(transcript.translation_languages)
    translated_transcript = transcript.translate(translate_lang)
    script_text = translated_transcript.fetch()
    text_formatter = TextFormatter()
    result_text = text_formatter.format_transcript(script_text)    
    return result_text

def word_count(self,string):
    return(len(string.strip().split(" ")))

if __name__=="__main__":
    langs={'Afrikaans': 'af', 'Akan': 'ak', 'Albanian': 'sq', 'Amharic': 'am', 'Arabic': 'ar', 'Armenian': 'hy', 'Assamese': 'as', 'Aymara': 'ay', 'Azerbaijani': 'az', 'Bangla': 'bn', 'Basque': 'eu', 'Belarusian': 'be', 'Bhojpuri': 'bho', 'Bosnian': 'bs', 'Bulgarian': 'bg', 'Burmese': 'my', 'Catalan': 'ca', 'Cebuano': 'ceb', 'Chinese (Simplified)': 'zh-Hans', 'Chinese (Traditional)': 'zh-Hant', 'Corsican': 'co', 'Croatian': 'hr', 'Czech': 'cs', 'Danish': 'da', 'Divehi': 'dv', 'Dutch': 'nl', 'English': 'en', 'Esperanto': 'eo', 'Estonian': 'et', 'Ewe': 'ee', 'Filipino': 'fil', 'Finnish': 'fi', 'French': 'fr', 'Galician': 'gl', 'Ganda': 'lg', 'Georgian': 'ka', 'German': 'de', 'Greek': 'el', 'Guarani': 'gn', 'Gujarati': 'gu', 'Haitian Creole': 'ht', 'Hausa': 'ha', 'Hawaiian': 'haw', 'Hebrew': 'iw', 'Hindi': 'hi', 'Hmong': 'hmn', 'Hungarian': 'hu', 'Icelandic': 'is', 'Igbo': 'ig', 'Indonesian': 'id', 'Irish': 'ga', 'Italian': 'it', 'Japanese': 'ja', 'Javanese': 'jv', 'Kannada': 'kn', 'Kazakh': 'kk', 'Khmer': 'km', 'Kinyarwanda': 'rw', 'Korean': 'ko', 'Krio': 'kri', 'Kurdish': 'ku', 'Kyrgyz': 'ky', 'Lao': 'lo', 'Latin': 'la', 'Latvian': 'lv', 'Lingala': 'ln', 'Lithuanian': 'lt', 'Luxembourgish': 'lb', 'Macedonian': 'mk', 'Malagasy': 'mg', 'Malay': 'ms', 'Malayalam': 'ml', 'Maltese': 'mt', 'Māori': 'mi', 'Marathi': 'mr', 'Mongolian': 'mn', 'Nepali': 'ne', 'Northern Sotho': 'nso', 'Norwegian': 'no', 'Nyanja': 'ny', 'Odia': 'or', 'Oromo': 'om', 'Pashto': 'ps', 'Persian': 'fa', 'Polish': 'pl', 'Portuguese': 'pt', 'Punjabi': 'pa', 'Quechua': 'qu', 'Romanian': 'ro', 'Russian': 'ru', 'Samoan': 'sm', 'Sanskrit': 'sa', 'Scottish Gaelic': 'gd', 'Serbian': 'sr', 'Shona': 'sn', 'Sindhi': 'sd', 'Sinhala': 'si', 'Slovak': 'sk', 'Slovenian': 'sl', 'Somali': 'so', 'Southern Sotho': 'st', 'Spanish': 'es', 'Sundanese': 'su', 'Swahili': 'sw', 'Swedish': 'sv', 'Tajik': 'tg', 'Tamil': 'ta', 'Tatar': 'tt', 'Telugu': 'te', 'Thai': 'th', 'Tigrinya': 'ti', 'Tsonga': 'ts', 'Turkish': 'tr', 'Turkmen': 'tk', 'Ukrainian': 'uk', 'Urdu': 'ur', 'Uyghur': 'ug', 'Uzbek': 'uz', 'Vietnamese': 'vi', 'Welsh': 'cy', 'Western Frisian': 'fy', 'Xhosa': 'xh', 'Yiddish': 'yi', 'Yoruba': 'yo', 'Zulu': 'zu'}
    audio_file="/home/pop/Music/mc_voice3.wav"
    #audio_file="/home/pop/Downloads/ted_60.wav"
    #https://www.youtube.com/watch?v=mu1Inz3ltlo (RAG)
    #https://www.youtube.com/watch?v=NePAPGxZnmE (Graph demo)
    # chatbot, history = faster_whisper(filepath=audio_file,task="transcribe")
    #chatbot, history = distil_whisper(filepath=audio_file)
    #result = download_youtube_transcript(video_id="8Wy5fqkQrI0")'Chinese (Simplified)'
    result = download_youtube_translate(video_id="8Wy5fqkQrI0",translate_lang="zh-Hans")
    print(langs['Akan'])
    #print(result)


