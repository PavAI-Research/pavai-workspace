import os
from dotenv import dotenv_values
config = {
    **dotenv_values("env.shared"),  # load shared development variables
    **dotenv_values("env.secret"),  # load sensitive variables
    **os.environ,  # override loaded values with environment variables
}

import gradio as gr
import pandas as pd
import os
import translator_ui
import scratchpad_ui
import chatbot_ui

# # theme=gr.themes.Monochrome()
# demoui = gr.Blocks(theme=gr.themes.Glass())

class AppMain(chatbot_ui.ChatbotWorkspace, scratchpad_ui.ScratchPad, translator_ui.CommunicationTranslator):

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

    def main(self):
        chatbot_tab = self.build_workspace_ui()
        translator_tab = self.build_translator_ui()                
        scratchpad_tab = self.build_scratchpad_ui()        
        self.app = gr.TabbedInterface(
            [chatbot_tab,translator_tab,scratchpad_tab],
            ["Chatbot At Work","Seamless Communication","Scratch Pad"],
            title="🌟✨ PAVAI Seamless Workspace",
             analytics_enabled=False 
        )

    def launch(self,server_name:str="0.0.0.0",server_port:int=7860,share:bool=False):
        background_image="resources/images/pavai_logo_large.png"
        authorized_users=[("abc:123"),("admin:123"),("john:smith"),("hello:hello")]      
        auth=[tuple(cred.split(':')) for cred in authorized_users] if authorized_users else None         
        absolute_path = os.path.abspath(background_image)
        self.startup_check()
        self.main()
        self.app.queue().launch(share=False,auth=None,allowed_paths=[absolute_path],server_name=server_name,server_port=server_port)


if __name__ == "__main__":
    app = AppMain()
    app.launch()
