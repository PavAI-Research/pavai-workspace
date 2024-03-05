import os
from dotenv import dotenv_values
config = {
    **dotenv_values("env.shared"),  # load shared development variables
    **dotenv_values("env.secret"),  # load sensitive variables
    **os.environ,  # override loaded values with environment variables
}

from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
import gradio as gr
from datetime import datetime
import pandas as pd
import os
import genai_at_work.translator_ui as translator_ui
import genai_at_work.scratchpad_ui as scratchpad_ui
import genai_at_work.chatbot_ui as chatbot_ui
import genai_at_work.summarizer_ui as summarizer_ui
import genai_at_work.websearch_ui as websearch_ui

# # theme=gr.themes.Monochrome()
# demoui = gr.Blocks(theme=gr.themes.Glass())

# create a FastAPI app
app = FastAPI()

# create a static directory to store the static files
static_dir = Path('./resources')
static_dir.mkdir(parents=True, exist_ok=True)

# mount FastAPI StaticFiles server
app.mount("/resources", StaticFiles(directory=static_dir), name="static")

class WorkspaceApp(chatbot_ui.ChatbotWorkspace, 
                   scratchpad_ui.ScratchPad, 
                   translator_ui.CommunicationTranslator, 
                   summarizer_ui.DocumentSummarizer, 
                   websearch_ui.WebResearchCrew):

    def __init__(self):
        super().__init__()

    def startup_check(self):
        ## check resources
        import download
        download.check_and_get_downloads()
        ## check config
        if not os.path.isfile("env.shared"):
            raise ValueError("Startup Failed! - missing env.shared config file!")
        ## check resource folder
        if not os.path.exists("resources"):
            raise ValueError("Startup Failed! - missing resources folder")
        ## check workspace folder
        if not os.path.exists("workspace"):
            os.makedirs("workspace/scratchpad")
            os.makedirs("workspace/downloads")            
            os.makedirs("workspace/session_logs")
        print("startup is good at this point")

    def main(self):
        chatbot_tab = self.build_workspace_ui()
        translator_tab = self.build_translator_ui()                
        scratchpad_tab = self.build_scratchpad_ui()  
        summarizer_tab = self.build_summarizer_ui()   
        websearch_tab = self.build_webresearch_ui()
        self.app = gr.TabbedInterface(
            [chatbot_tab,translator_tab,summarizer_tab,websearch_tab,scratchpad_tab],
            ["Chatbot At Work","Seamless Communication","Long-Text Summarizer","Web Search Crew","Scratch Pad"],
            title="🌟✨ Pavai Seamless Workspace",
             analytics_enabled=False 
        )
        print("system ready!")

    def launch(self,server_name:str="0.0.0.0",server_port:int=7860,share:bool=False):
        background_image="resources/images/pavai_logo_large.png"
        authorized_users=[("abc:123"),("john:smith"),("pavai:pavai")]      
        auth=[tuple(cred.split(':')) for cred in authorized_users] if authorized_users else None         
        absolute_path = os.path.abspath(background_image)
        self.startup_check()
        self.main()
        return self.app
        #self.app.queue().launch(share=False,auth=None,allowed_paths=[absolute_path],server_name=server_name,server_port=server_port)

workspaceblock=WorkspaceApp()
#voiceblock.launch(server_name=server_name,server_port=server_port,share=share)

# mount Gradio app to FastAPI app
background_image="resources/images/pavai_logo_large.png"
absolute_path = os.path.abspath(background_image)
app = gr.mount_gradio_app(app, workspaceblock.launch(), path="/")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
    #app = AppMain()
    #app.launch()
