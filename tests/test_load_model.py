import genai_at_work.translator as translator 
import genai_at_work.llamacpp as llamacpp 
import genai_at_work.mediadata as mediadata

if __name__=="__main__":
    model_file, project_file = llamacpp.download_models()
    print(model_file)
    print(project_file)

    test_audio_filepath="/home/pop/software_engineer/GenAI-At-Work/samples/samples_jfk.wav"
    audio, out_text= translator.run_s2st(input_audio_filepath=test_audio_filepath,
                                         source_language="English",
                                         target_language="Spanish")
    print(out_text)

    chatbot, history = mediadata.faster_whisper(filepath=test_audio_filepath,task="transcribe")
    print(chatbot)    
    chatbot, history = mediadata.distil_whisper(filepath=test_audio_filepath)
    print(chatbot)
