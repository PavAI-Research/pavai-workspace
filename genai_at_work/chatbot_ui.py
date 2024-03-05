from genai_at_work import config, logutil
logger = logutil.logging.getLogger(__name__)

import gradio as gr
import pandas as pd
import os
import time
#import chatopenai
import genai_at_work.chatbotllm as chatbotllm
import genai_at_work.chatprompt as chatprompt
import genai_at_work.mediadata as mediadata
import genai_at_work.filedata as filedata
import genai_at_work.webdata as webdata
import genai_at_work.historydata as historydata

class ChatbotWorkspace:

    def datasource_selection(self, ds):
        print(f"selected {ds}")
        if ds == "image":
            return {
                self.upload_youtube_url: gr.Text(),
                self.upload_web_url: gr.Text(),
                self.upload_web_search: gr.Text(),
                self.upload_image_box: gr.Image(visible=True, interactive=True),
                self.upload_audio_box: gr.Audio(),
                self.upload_video_box: gr.Video(),
                self.upload_file_box: gr.File(),
            }
        elif ds == "video":
            return {
                self.upload_youtube_url: gr.Text(),
                self.upload_web_url: gr.Text(),
                self.upload_web_search: gr.Text(),
                self.upload_image_box: gr.Image(),
                self.upload_audio_box: gr.Audio(),
                self.upload_video_box: gr.Video(visible=True, interactive=True),
                self.upload_file_box: gr.File(),
            }
        elif ds == "audio":
            return {
                self.upload_youtube_url: gr.Text(),
                self.upload_web_url: gr.Text(),
                self.upload_web_search: gr.Text(),
                self.upload_image_box: gr.Image(),
                self.upload_audio_box: gr.Audio(visible=True, interactive=True),
                self.upload_video_box: gr.Video(),
                self.upload_file_box: gr.File(),
            }
        elif ds == "file":
            return {
                self.upload_youtube_url: gr.Text(),
                self.upload_web_url: gr.Text(),
                self.upload_web_search: gr.Text(),
                self.upload_image_box: gr.Image(),
                self.upload_audio_box: gr.Audio(),
                self.upload_video_box: gr.Video(),
                self.upload_file_box: gr.File(visible=True, interactive=True),
            }
        elif ds == "youtube-url":
            return {
                self.upload_youtube_url: gr.Text(visible=True, interactive=True),
                self.upload_web_url: gr.Text(),
                self.upload_web_search: gr.Text(),
                self.upload_image_box: gr.Image(),
                self.upload_audio_box: gr.Audio(),
                self.upload_video_box: gr.Video(),
                self.upload_file_box: gr.File(),
            }
        elif ds == "web-url":
            return {
                self.upload_youtube_url: gr.Text(),
                self.upload_web_url: gr.Text(visible=True, interactive=True),
                self.upload_web_search: gr.Text(),
                self.upload_image_box: gr.Image(),
                self.upload_audio_box: gr.Audio(),
                self.upload_video_box: gr.Video(),
                self.upload_file_box: gr.File(),
            }
        elif ds == "web-search":
            return {
                self.upload_youtube_url: gr.Text(),
                self.upload_web_url: gr.Text(),
                self.upload_web_search: gr.Text(visible=True, interactive=True),
                self.upload_image_box: gr.Image(),
                self.upload_audio_box: gr.Audio(),
                self.upload_video_box: gr.Video(),
                self.upload_file_box: gr.File(),
            }
        else:
            return {
                self.upload_youtube_url: gr.Text(),
                self.upload_web_url: gr.Text(),
                self.upload_web_search: gr.Text(),
                self.upload_image_box: gr.Image(),
                self.upload_audio_box: gr.Audio(),
                self.upload_video_box: gr.Video(),
                self.upload_file_box: gr.File(),
            }

    def show_history(self, history_df: gr.DataFrame, colname: str = None):
        col = historydata.get_collection(colname)
        result = historydata.peek_collection(colname)
        i = 0
        history_df = pd.DataFrame(columns=["id", "content"])
        for i in range(len(result)):
            new_row = {"id": result[i][0], "content": result[i][1]}
            history_df.loc[len(history_df)] = new_row
        return history_df, col.count()

    def search_history(
        self, history_df: gr.DataFrame, colname: str = None, query: str = None
    ):
        result = historydata.query_collection(colname=colname, query=query)
        i = 0
        history_df = pd.DataFrame(columns=["id", "content"])
        for i in range(len(result)):
            new_row = {"id": result[i][0], "content": result[i][1]}
            history_df.loc[len(history_df)] = new_row
        return history_df

    def reset_history(self):
        gr.Warning("resetting history. Please testart the app.")
        historydata.reset()

    ## in-line update event handler
    def update_expert_model(
        self, expert_mode=None, system_prompt=None, speech_style=None
    ):
        if expert_mode is None:
            return
        system_prompt = chatprompt.domain_experts[expert_mode]
        gr.Info(f"set expert mode {expert_mode}\n{system_prompt}")
        if speech_style is not None and len(speech_style) > 0:
            system_prompt = (
                system_prompt
                + f"\n You response style should be in {speech_style} style."
            )
        return gr.Textbox(value=system_prompt, interactive=True)

    def update_speech_style(
        self, expert_mode=None, system_prompt=None, speech_style=None
    ):
        gr.Info(f"switch speech_style to {speech_style}")
        if expert_mode is not None and len(expert_mode) > 0:
            system_prompt = chatprompt.domain_experts[expert_mode]
        if system_prompt is None:
            system_prompt = f"\nYou response should be in {speech_style} style."
        if speech_style is not None and len(speech_style) > 0:
            system_prompt = (
                system_prompt + f"\nYou response should be in {speech_style} style."
            )
        return gr.Textbox(value=system_prompt, interactive=True)

    def update_active_model(self, new_model):
        gr.Info(f"switch model to {new_model}")
        return gr.Textbox(value=new_model, interactive=True)

    def load_chat_session(self, chatlogfile, chatbot, state):
        if chatlogfile is None:
            gr.Warning(f"no session log selected")
            return
        chatbot, state = filedata.load_session_files(chatlogfile, chatbot, state)
        gr.Info(f"loaded session log {chatlogfile}")
        return chatbot, state

    def save_chat_session(self, chatbot, state):
        save_chatbot_file = filedata.save_session_files(chatbot, state)
        outputs = filedata.list_session_files()
        gr.Info(f"saved session {save_chatbot_file}")
        return outputs

    def delete_chat_session(self, chatlogfile):
        if chatlogfile is None:
            gr.Warning(f"no session log selected")
            return
        filedata.delete_session_files(chatlogfile)
        outputs = filedata.list_session_files()
        gr.Info(f"deleted session log {chatlogfile}")
        return outputs

    def transcribe_audio(
        self,
        audio_file,
        chatbot: list,
        history: list,
        model_type: str = "distil_whisper",
    ):
        gr.Info("transcribing audio, please wait!")
        if model_type == "distil_whisper":
            chatbot, history = mediadata.distil_whisper(
                filepath=audio_file, chatbot=chatbot, history=history
            )
        elif model_type == "faster_whisper":
            chatbot, history = mediadata.faster_whisper(
                filepath=audio_file, chatbot=chatbot, history=history
            )
        else:
            raise ValueError(f"unsupported model type {model_type}")

        if self.ckb_save_result.value == True:
            historydata.add_collection_record(
                colname="upload_audio_collection",
                weburl=audio_file,
                filecontent=chatbot[-1][1],
            )

        return chatbot, history

    def transcribe_video(
        self,
        audio_file,
        chatbot: list,
        history: list,
        model_type: str = "distil_whisper",
    ):
        gr.Info("transcribing video, please wait!")
        chatbot, history = mediadata.transcribe_video(
            filepath=audio_file,
            chatbot=chatbot,
            history=history,
            export_transcriber=model_type,
        )
        if self.ckb_save_result.value == True:
            historydata.add_collection_record(
                colname="upload_video_collection",
                weburl=audio_file,
                filecontent=chatbot[-1][1],
            )

        return chatbot, history

    def transcribe_youtube(
        self, fileurl, chatbot: list, history: list, model_type: str = "distil_whisper"
    ):
        gr.Info("transcribing youtube video, please wait!")
        chatbot, history = mediadata.transcribe_youtube(
            input_url=fileurl,
            chatbot=chatbot,
            history=history,
            export_transcriber=model_type,
        )
        if self.ckb_save_result.value == True:
            historydata.add_collection_record(
                colname="youtube_video_collection",
                weburl=fileurl,
                filecontent=chatbot[-1][1],
            )

        return chatbot, history

    def scrape_web(
        self, website_url, chatbot: list, history: list, tool: str = "newspaper"
    ):
        gr.Info("scrapping web page, please wait!")
        chatbot, history = webdata.scrap_web(
            website_url=website_url, chatbot=chatbot, history=history, tool=tool
        )
        if self.ckb_save_result.value == True:
            historydata.add_collection_record(
                colname="web_scrapping_collection",
                weburl=website_url,
                filecontent=chatbot[-1][1],
            )

        return chatbot, history

    def web_search(
        self, keywords, chatbot: list, history: list, tool: str = "duckduckgo"
    ):
        gr.Info("searching web, please wait!")
        chatbot, history = webdata.web_search(
            keywords=keywords, chatbot=chatbot, history=history, tool=tool
        )
        if self.ckb_save_result.value == True:
            historydata.add_collection_record(
                colname="web_search_collection",
                weburl=keywords,
                filecontent=chatbot[-1][1],
            )
        return chatbot, history

    def get_api_host(self):
        if config.config["LLM_PROVIDER"]=="all-in-one":
            return config.config["LLM_PROVIDER"]                   
        elif config.config["LLM_PROVIDER"]=="ollama":
            return config.config["OLLAMA_API_HOST"]
        elif config.config["LLM_PROVIDER"]=="openai":    
            return config.config["OPENAI_API_HOST"]           
        else:
            return "Missing, configuration error!"

    def build_workspace_ui(self):
        self.blocks_workspace = gr.Blocks(analytics_enabled=False)
        with self.blocks_workspace:
            state = gr.State()
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Accordion("Web Tools", open=False):
                        web_search_tool = gr.Radio(
                            label="Web Search",
                            choices=["duckduckgo"],
                            value="duckduckgo",
                            interactive=True,
                        )
                        web_scrapping_tool = gr.Radio(
                            label="Web Scrapping",
                            choices=["newspaper", "selenium"],
                            value="newspaper",
                            interactive=True,
                        )
                    datasource_dropdown = gr.Dropdown(
                        choices=[
                            "web-search",
                            "web-url",
                            "youtube-url",
                            "file",
                            "image",
                            "video",
                            "audio",
                        ],
                        label="Data Source",
                        info="specify type of data source include in the chat",
                        interactive=True,
                    )
                    self.upload_youtube_url = gr.Text(
                        label="enter youtube url",
                        visible=False,
                        interactive=True,
                        info="press Ctl+Enter key to submit",
                    )
                    self.upload_web_url = gr.Text(
                        label="enter web url",
                        visible=False,
                        interactive=True,
                        info="press Ctl+Enter key to submit",
                    )
                    self.upload_web_search = gr.Text(
                        label="enter web search keywords",
                        visible=False,
                        interactive=True,
                        info="press Ctl+Enter key to submit",
                    )
                    self.upload_image_box = gr.Image(
                        type="filepath", visible=False, interactive=True
                    )
                    self.upload_audio_box = gr.Audio(
                        type="filepath", visible=False, interactive=True
                    )
                    self.upload_video_box = gr.Video(
                        value=None, visible=False, interactive=True
                    )
                    self.upload_file_box = gr.File(
                        type="filepath", visible=False, interactive=True
                    )

                    ## input functions
                    datasource_dropdown.change(
                        fn=self.datasource_selection,
                        inputs=datasource_dropdown,
                        outputs=[
                            self.upload_youtube_url,
                            self.upload_web_url,
                            self.upload_web_search,
                            self.upload_image_box,
                            self.upload_audio_box,
                            self.upload_video_box,
                            self.upload_file_box,
                        ],
                    )

                    with gr.Row():
                        with gr.Column(scale=1):
                            self.ckb_save_result = gr.Checkbox(
                                value=True,
                                label="save result history",
                                info="Save result data to local history db",
                            )

                    with gr.Accordion("History", open=False):
                        gr.Markdown(" ")
                        history_cols = gr.Dropdown(
                            label="data collection",
                            choices=historydata.list_collections(),
                            info="select a history collection to view or query",
                        )
                        collection_records = gr.Number(
                            label="total documents", interactive=False
                        )
                        history_collection_search = gr.Text(
                            label="search collection",
                            visible=True,
                            interactive=True,
                            info="press Ctl+Enter key to submit",
                        )
                        history_document_id = gr.Text(
                            label="load to chat",
                            visible=True,
                            interactive=True,
                            info="Enter ID then press Ctl+Enter key to submit",
                        )
                        ## session data
                        history_df = gr.DataFrame(
                            label="Documents",
                            headers=["id", "content"],
                            datatype=["str", "str"],
                            col_count=(2, "fixed"),
                            interactive=False,
                        )

                        history_cols.change(
                            fn=self.show_history,
                            inputs=[history_df, history_cols],
                            outputs=[history_df, collection_records],
                        )
                        history_collection_search.submit(
                            fn=self.search_history,
                            inputs=[
                                history_df,
                                history_cols,
                                history_collection_search,
                            ],
                            outputs=[history_df],
                        )

                        btn_reset_history = gr.Button("Reset history", size="sm")
                        btn_reset_history.click(fn=self.reset_history)

                with gr.Column(scale=7):
                    ## CHATBOT ##
                    with gr.Accordion("Generative AI Chatbot", open=True):
                        with gr.Accordion("Assistant Mode", open=False):
                            with gr.Row():
                                with gr.Column():
                                    expert_mode = gr.Dropdown(
                                        label="Domain Expert:",
                                        choices=chatprompt.domain_experts.keys(),
                                        interactive=True,
                                    )
                                with gr.Column():
                                    speech_style = gr.Dropdown(
                                        label="Speech and Writing Style:",
                                        choices=chatprompt.speech_styles.keys(),
                                        interactive=True,
                                    )

                        # with gr.Accordion("Session Logs", open=False):
                        #     with gr.Row():
                        #         chat_logs = gr.Dropdown(
                        #             label="Log files",
                        #             choices=filedata.list_session_files(),
                        #             interactive=True,
                        #         )
                        #     with gr.Row():
                        #         btn_load_chat_log = gr.Button(
                        #             value="Load to chat",
                        #             size="sm",
                        #             scale=2,
                        #             min_width=30,
                        #             interactive=True,
                        #         )
                        #         btn_delete_chat_log = gr.Button(
                        #             value="Delete",
                        #             size="sm",
                        #             scale=1,
                        #             min_width=30,
                        #             interactive=True,
                        #         )

                        chatbot = gr.Chatbot(
                            [],
                            label="Chatbot",
                            elem_id="chatbot",
                            bubble_full_width=False,
                            show_copy_button=True
                        )
                        with gr.Row():
                            with gr.Column(scale=2):
                                gr.HTML()
                            with gr.Column(scale=2):
                                server_status_code = gr.HTML(
                                    label="api finish status: "
                                )

                        with gr.Row():
                            box_message = gr.Textbox(
                                scale=5,
                                show_label=False,
                                placeholder="Enter text and press enter",
                                container=False,
                                autofocus=True,
                            )
                        with gr.Row():
                            btn_submit = gr.Button("Send", scale=3)
                            btn_clear = gr.ClearButton(
                                [
                                    speech_style,
                                    expert_mode,
                                    box_message,
                                    chatbot,
                                    self.upload_youtube_url,
                                    self.upload_web_url,
                                    self.upload_web_search,
                                    self.upload_image_box,
                                    self.upload_audio_box,
                                    self.upload_video_box,
                                    self.upload_file_box,
                                    state,
                                ],
                                value="Clear & New",
                                size="sm",
                                scale=1,
                            )
                            btn_save_chat = gr.Button(
                                value="Save Session",
                                size="sm",
                                scale=1,
                                min_width=30,
                                interactive=True,
                            )
                                                             
                        with gr.Accordion("Model Parameters", open=False):
                            with gr.Row():
                                with gr.Column():
                                    api_host = gr.Textbox(
                                        label="API base",
                                        value=self.get_api_host(),
                                    )
                                with gr.Column():
                                    active_model = gr.Textbox(
                                        label="Active model",
                                        value="llava:7b-v1.6-mistral-q5_0",
                                        info="current model",
                                    )
                                with gr.Column():
                                    api_key = gr.Textbox(
                                        label="API Key",
                                        value="ollama",
                                        type="password",
                                        placeholder="sk..",
                                        info="You have to provide your own GPT4 keys for this app to function properly",
                                    )
                                with gr.Column():
                                    model_dropdown = gr.Dropdown(
                                        label="Available models",
                                        value="zephyr:latest",
                                        choices=chatbotllm.list_models(),
                                        info="select a model",
                                        interactive=True,
                                    )

                            ## in-line update event handler
                            model_dropdown.change(
                                fn=self.update_active_model,
                                inputs=[model_dropdown],
                                outputs=[active_model],
                            )

                            system_msg_info = """System message helps set the behavior of the AI Assistant. For example, the assistant could be instructed with 'You are a helpful assistant.'"""
                            system_prompt = gr.Textbox(
                                label="Instruct the AI Assistant to set its beaviour",
                                info=system_msg_info,
                                value="You are helpful AI assistant on helping answer user question and research.",
                                placeholder="Type here..",
                                lines=2,
                            )
                            accordion_msg = gr.HTML(
                                value="🚧 To set System message you will have to refresh the app",
                                visible=False,
                            )
                            # top_p, temperature
                            with gr.Row():
                                with gr.Column():
                                    top_p = gr.Slider(
                                        minimum=-0,
                                        maximum=40.0,
                                        value=1.0,
                                        step=0.05,
                                        interactive=True,
                                        label="Top-p (nucleus sampling)",
                                    )
                                with gr.Column():
                                    temperature = gr.Slider(
                                        minimum=0,
                                        maximum=5.0,
                                        value=1.0,
                                        step=0.1,
                                        interactive=True,
                                        label="Temperature",
                                    )
                            with gr.Row():
                                with gr.Column():
                                    max_tokens = gr.Slider(
                                        minimum=1,
                                        maximum=16384,
                                        value=1024,
                                        step=1,
                                        interactive=True,
                                        label="Max Tokens",
                                    )
                                with gr.Column():
                                    presence_penalty = gr.Number(
                                        label="presence_penalty", value=0, precision=0
                                    )
                                with gr.Column():
                                    stop_words = gr.Textbox(
                                        label="stop words", value="<"
                                    )
                                with gr.Column():
                                    frequency_penalty = gr.Number(
                                        label="frequency_penalty", value=0, precision=0
                                    )
                                with gr.Column():
                                    chat_counter = gr.Number(
                                        value=0, visible=False, precision=0
                                    )

                        with gr.Accordion("Session Logs", open=False):
                            with gr.Row():
                                chat_logs = gr.Dropdown(
                                    label="Log files",
                                    choices=filedata.list_session_files(),
                                    interactive=True,
                                )
                            with gr.Row():
                                btn_load_chat_log = gr.Button(
                                    value="Load to chat",
                                    size="sm",
                                    scale=2,
                                    min_width=30,
                                    interactive=True,
                                )
                                btn_delete_chat_log = gr.Button(
                                    value="Delete",
                                    size="sm",
                                    scale=1,
                                    min_width=30,
                                    interactive=True,
                                )

                    with gr.Accordion("Scratch Pad", open=False):
                        box_notepad=gr.TextArea(lines=7,label="Notes", info="write single page quick notes.", interactive=True)                    
                        with gr.Row():
                            with gr.Column(scale=3):
                                btn_save_notepad=gr.Button(size="sm",value="save and update")
                            with gr.Column(scale=1):                            
                                btn_load_notepad=gr.Button(size="sm",value="load")                    
                        ## function: scratch pad
                        def save_notespad(text_notes):
                            gr.Info("saved notes to notepad.txt!")
                            return filedata.save_text_file("workspace/scratchpad/notepad.txt",text_notes)
                        def load_notespad(filename:str="workspace/scratchpad/notepad.txt"):
                            gr.Info("load notes")                    
                            return filedata.get_text_file(filename)

                        btn_save_notepad.click(fn=save_notespad, inputs=[box_notepad])
                        btn_load_notepad.click(fn=load_notespad, outputs=[box_notepad])                

                    ## update expert mode and speech style
                    ## in-line update event handler                        
                    expert_mode.change(
                        fn=self.update_expert_model,
                        inputs=[expert_mode, system_prompt, speech_style],
                        outputs=[system_prompt],
                    )
                    speech_style.change(
                        fn=self.update_speech_style,
                        inputs=[expert_mode, system_prompt, speech_style],
                        outputs=[system_prompt],
                    )

                    ## input message
                    txt_message = box_message.submit(
                        chatbotllm.message_and_history_v2,
                        inputs=[
                            api_host,
                            api_key,
                            active_model,
                            box_message,
                            chatbot,
                            state,
                            system_prompt,
                            top_p,
                            max_tokens,
                            temperature,
                            stop_words,
                            presence_penalty,
                            frequency_penalty,
                        ],
                        outputs=[chatbot, state, server_status_code],
                        queue=False,
                    )
                    txt_message.then(
                        lambda: gr.Textbox(value="", interactive=True),
                        None,
                        [box_message],
                        queue=False,
                    )
                    ## send button
                    btn_clicked = btn_submit.click(
                        chatbotllm.message_and_history_v2,
                        inputs=[
                            api_host,
                            api_key,
                            active_model,
                            box_message,
                            chatbot,
                            state,
                            system_prompt,
                            top_p,
                            max_tokens,
                            temperature,
                            stop_words,
                            presence_penalty,
                            frequency_penalty,
                        ],
                        outputs=[chatbot, state, server_status_code],
                    )
                    btn_clicked.then(
                        lambda: gr.Textbox(value="", interactive=True),
                        None,
                        [box_message],
                        queue=False,
                    )

                    ## session logs buttons
                    btn_save_chat.click(
                        fn=self.save_chat_session,
                        inputs=[chatbot, state],
                        outputs=[chat_logs],
                    )
                    btn_delete_chat_log.click(
                        fn=self.delete_chat_session,
                        inputs=[chat_logs],
                        outputs=[chat_logs],
                    )
                    btn_load_chat_log.click(
                        fn=self.load_chat_session,
                        inputs=[chat_logs, chatbot, state],
                        outputs=[chatbot, state],
                    )

                ## upload image ##
                self.upload_image_box.upload(
                    fn=chatbotllm.upload_image,
                    inputs=[
                        api_host,
                        api_key,
                        active_model,
                        box_message,
                        self.upload_image_box,
                        chatbot,
                        state,
                        system_prompt,
                    ],
                    outputs=[chatbot, state],
                )

                self.upload_audio_box.upload(
                    fn=self.transcribe_audio,
                    inputs=[self.upload_audio_box, chatbot, state],
                    outputs=[chatbot, state],
                )

                self.upload_video_box.upload(
                    fn=self.transcribe_video,
                    inputs=[self.upload_video_box, chatbot, state],
                    outputs=[chatbot, state],
                )
                self.upload_file_box.upload(
                    fn=filedata.load_file_content,
                    inputs=[self.upload_file_box, chatbot, state],
                    outputs=[chatbot, state],
                )

                self.upload_youtube_url.submit(
                    fn=self.transcribe_youtube,
                    inputs=[self.upload_youtube_url, chatbot, state],
                    outputs=[chatbot, state],
                )
                self.upload_web_url.submit(
                    fn=self.scrape_web,
                    inputs=[self.upload_web_url, chatbot, state, web_scrapping_tool],
                    outputs=[chatbot, state],
                )

                self.upload_web_search.submit(
                    fn=self.web_search,
                    inputs=[self.upload_web_search, chatbot, state, web_search_tool],
                    outputs=[chatbot, state],
                )

                history_document_id.submit(
                    fn=historydata.get_document,
                    inputs=[history_cols, history_document_id, chatbot, state],
                    outputs=[chatbot, state],
                )
        return self.blocks_workspace
