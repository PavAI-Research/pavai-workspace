from rich.console import Console
from rich.progress import Progress

import genai_at_work.chatllamacpp as chatllamacpp
import genai_at_work.multilingual as multilingual

def main():
    with Progress(transient=True) as progress: 
        try:       
            task = progress.add_task("checking system resources...", total=2)

            print("1.Download Local LLM Model")
            chatllamacpp.LlamaCppLocal().download_default_model()
            print("1.completed")                
            progress.advance(task)

            print("2.Download translator file")
            multilingual.SeamlessM4T().load_model()        
            print("2.completed")                
            progress.advance(task)
        except Exception as e:
            print("System setup error ocurred")
            print(e)

if __name__=="__main__":
    print("First Time System Setup --- Begin ")
    main()
    print("First Time System Setup --- Finish ")
