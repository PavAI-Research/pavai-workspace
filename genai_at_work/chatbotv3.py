import gradio as gr
import pandas as pd
import os
import time
import chatopenai
import chatprompt
import mediadata
import filedata
import webdata
import historydata
import translator

# theme=gr.themes.Monochrome()
demoui = gr.Blocks(theme=gr.themes.Glass(), title="Seamless Workspace")
with demoui:
    state = gr.State()
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("Web Tools", open=False):
                web_search_tool = gr.Radio(label="Web Search",choices=["duckduckgo"],value="duckduckgo", interactive=True)                        
                web_scrapping_tool = gr.Radio(label="Web Scrapping",choices=["newspaper","selenium"],value="newspaper", interactive=True)            
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
            upload_youtube_url = gr.Text(
                label="enter youtube url", visible=False, interactive=True, info="press Ctl+Enter key to submit"
            )
            upload_web_url = gr.Text(
                label="enter web url", visible=False, interactive=True, info="press Ctl+Enter key to submit"
            )
            upload_web_search = gr.Text(
                label="enter web search keywords", visible=False, interactive=True, info="press Ctl+Enter key to submit"
            )
            upload_image_box = gr.Image(
                type="filepath", visible=False, interactive=True
            )
            upload_audio_box = gr.Audio(
                type="filepath", visible=False, interactive=True
            )
            upload_video_box = gr.Video(value=None, visible=False, interactive=True)
            upload_file_box = gr.File(type="filepath", visible=False, interactive=True)

            ## input functions
            def datasource_selection(ds):
                print(f"selected {ds}")
                if ds == "image":
                    return {
                        upload_youtube_url: gr.Text(),
                        upload_web_url: gr.Text(),
                        upload_web_search: gr.Text(),
                        upload_image_box: gr.Image(visible=True, interactive=True),
                        upload_audio_box: gr.Audio(),
                        upload_video_box: gr.Video(),
                        upload_file_box: gr.File(),
                    }
                elif ds == "video":
                    return {
                        upload_youtube_url: gr.Text(),
                        upload_web_url: gr.Text(),
                        upload_web_search: gr.Text(),
                        upload_image_box: gr.Image(),
                        upload_audio_box: gr.Audio(),
                        upload_video_box: gr.Video(visible=True, interactive=True),
                        upload_file_box: gr.File(),
                    }
                elif ds == "audio":
                    return {
                        upload_youtube_url: gr.Text(),
                        upload_web_url: gr.Text(),
                        upload_web_search: gr.Text(),
                        upload_image_box: gr.Image(),
                        upload_audio_box: gr.Audio(visible=True, interactive=True),
                        upload_video_box: gr.Video(),
                        upload_file_box: gr.File(),
                    }
                elif ds == "file":
                    return {
                        upload_youtube_url: gr.Text(),
                        upload_web_url: gr.Text(),
                        upload_web_search: gr.Text(),
                        upload_image_box: gr.Image(),
                        upload_audio_box: gr.Audio(),
                        upload_video_box: gr.Video(),
                        upload_file_box: gr.File(visible=True, interactive=True),
                    }
                elif ds == "youtube-url":
                    return {
                        upload_youtube_url: gr.Text(visible=True, interactive=True),
                        upload_web_url: gr.Text(),
                        upload_web_search: gr.Text(),
                        upload_image_box: gr.Image(),
                        upload_audio_box: gr.Audio(),
                        upload_video_box: gr.Video(),
                        upload_file_box: gr.File(),
                    }
                elif ds == "web-url":
                    return {
                        upload_youtube_url: gr.Text(),
                        upload_web_url: gr.Text(visible=True, interactive=True),
                        upload_web_search: gr.Text(),
                        upload_image_box: gr.Image(),
                        upload_audio_box: gr.Audio(),
                        upload_video_box: gr.Video(),
                        upload_file_box: gr.File(),
                    }
                elif ds == "web-search":
                    return {
                        upload_youtube_url: gr.Text(),
                        upload_web_url: gr.Text(),
                        upload_web_search: gr.Text(visible=True, interactive=True),
                        upload_image_box: gr.Image(),
                        upload_audio_box: gr.Audio(),
                        upload_video_box: gr.Video(),
                        upload_file_box: gr.File(),
                    }
                else:
                    return {
                        upload_youtube_url: gr.Text(),
                        upload_web_url: gr.Text(),
                        upload_web_search: gr.Text(),
                        upload_image_box: gr.Image(),
                        upload_audio_box: gr.Audio(),
                        upload_video_box: gr.Video(),
                        upload_file_box: gr.File(),
                    }

            datasource_dropdown.change(
                fn=datasource_selection,
                inputs=datasource_dropdown,
                outputs=[
                    upload_youtube_url,
                    upload_web_url,
                    upload_web_search,
                    upload_image_box,
                    upload_audio_box,
                    upload_video_box,
                    upload_file_box,
                ],
            )
            # styler = df.style.highlight_max(color = 'lightgreen', axis = 0)

            with gr.Row():
                with gr.Column(scale=1):
                    ckb_save_result=gr.Checkbox(value=True,label="save result history",info="Save result data to local history db")            

            with gr.Accordion("History", open=False):
                gr.Markdown(" ")
                history_cols=gr.Dropdown(label="data collection",choices=historydata.list_collections(),info="select a history collection to view or query")
                collection_records=gr.Number(label="total documents",interactive=False)
                history_collection_search = gr.Text(
                    label="search collection", visible=True, interactive=True, info="press Ctl+Enter key to submit"
                )
                history_document_id = gr.Text(
                    label="load to chat", visible=True, interactive=True, info="Enter ID then press Ctl+Enter key to submit"
                )                 
                ## session data
                history_df = gr.DataFrame(
                    label="Documents", 
                    headers=["id", "content"],
                    datatype=["str", "str"],
                    col_count=(2, "fixed"),
                    interactive=False,
                )
                def show_history(history_df:gr.DataFrame,colname: str=None):
                    col = historydata.get_collection(colname)
                    result = historydata.peek_collection(colname)
                    i=0
                    history_df=pd.DataFrame(columns=['id', 'content'])
                    for i in range(len(result)):
                        new_row = {"id": result[i][0], "content": result[i][1]}
                        history_df.loc[len(history_df)] = new_row
                    return history_df,col.count()  
                
                def search_history(history_df:gr.DataFrame,colname: str=None, query:str=None):
                    result = historydata.query_collection(colname=colname,query=query)
                    i=0
                    history_df=pd.DataFrame(columns=['id', 'content'])
                    for i in range(len(result)):
                        new_row = {"id": result[i][0], "content": result[i][1]}
                        history_df.loc[len(history_df)] = new_row
                    return history_df

                history_cols.change(fn=show_history, inputs=[history_df,history_cols], outputs=[history_df,collection_records])
                history_collection_search.submit(fn=search_history,inputs=[history_df,history_cols,history_collection_search], outputs=[history_df])    

                btn_reset_history = gr.Button("Reset history",size="sm")

                def reset_history():
                    gr.Warning("resetting history. Please testart the app.")
                    historydata.reset()
                btn_reset_history.click(fn=reset_history)
                ## functions

        with gr.Column(scale=7):
            ## Seamless Communication ##
            with gr.Accordion("Realtime Speech-to-Speech Translator", open=False):
                gr.Markdown(
                    "Speak on your own native language over-voice with another party using realtime speech-to-speech translation."
                )
                with gr.Group(visible=True) as voicechat:
                    with gr.Row():
                        with gr.Column(scale=5):
                            source_language = gr.Dropdown(
                                label="User Spoken language",
                                choices=translator.ASR_TARGET_LANGUAGE_NAMES,
                                value="English"
                            )
                            input_audio = gr.Audio(
                                sources=["microphone","upload"],
                                label="User voice",
                                type="filepath",
                                scale=1,
                                min_width=20,
                            )
                            #user_speech_text = gr.Text(label="Your voice text", lines=2)
                            btn_translate = gr.Button("Translate User Speech to Party", size="sm", scale=1)
                        with gr.Column(scale=5):
                            party_source_language = gr.Dropdown(
                                label="Party Spoken language",
                                choices=translator.ASR_TARGET_LANGUAGE_NAMES,
                                value="French",
                            )
                            party_input_audio = gr.Audio(
                                sources=["microphone","upload"],                                
                                label="Party voice",
                                type="filepath",
                                scale=1,
                                min_width=20,
                            )
                            #party_speech_text = gr.Text(label="Party voice text", lines=2)
                            party_btn_translate = gr.Button("Translate Party Speech to User", size="sm", scale=1)

                    with gr.Row():           
                        with gr.Column():
                            with gr.Group():
                                target_language = gr.Dropdown(
                                    label="Target Party language",
                                    choices=translator.S2ST_TARGET_LANGUAGE_NAMES,
                                    value=translator.DEFAULT_TARGET_LANGUAGE,
                                )
                                xspeaker = gr.Slider(1, 100, value=7,step=1,label="Speaker Id", interactive=True)
                                output_audio = gr.Audio(
                                    label="Translated speech",
                                    autoplay=True,
                                    streaming=False,
                                    type="numpy",
                                )
                                output_text = gr.Textbox(label="Translated text", lines=3)
                                btn_clear = gr.ClearButton([source_language,input_audio,output_audio, output_text], size="sm")

                        ## Other Party
                        with gr.Column():
                            with gr.Group():
                                party_target_language = gr.Dropdown(
                                    label="Target User language",
                                    choices=translator.S2ST_TARGET_LANGUAGE_NAMES,
                                    value="English",
                                )
                                party_xspeaker = gr.Slider(1, 100, value=7,step=1,label="Speaker Id", interactive=True)
                                party_output_audio = gr.Audio(
                                    label="Translated speech",
                                    autoplay=True,
                                    streaming=False,
                                    type="numpy",
                                )
                                party_output_text = gr.Textbox(label="Translated text", lines=3)
                                party_btn_clear = gr.ClearButton([party_source_language,party_input_audio,party_output_audio, party_output_text],size="sm")

                                # handle speaker id change
                                party_xspeaker.change(fn=translator.update_value, inputs=xspeaker)
                        # handle
                        btn_translate.click(
                            fn=translator.run_s2st,
                            inputs=[input_audio, source_language, target_language,xspeaker],
                            outputs=[output_audio, output_text],
                            api_name="s2st",
                        )

                        # handle
                        party_btn_translate.click(
                            fn=translator.run_s2st,
                            inputs=[party_input_audio, party_source_language, party_target_language,party_xspeaker],
                            outputs=[party_output_audio, party_output_text],
                            api_name="s2st_party",
                        )

                        ## auto submit
                        input_audio.stop_recording(
                            fn=translator.run_s2st,
                            inputs=[input_audio, source_language, target_language],
                            outputs=[output_audio, output_text],
                        )

                        ## auto submit
                        party_input_audio.stop_recording(
                            fn=translator.run_s2st,
                            inputs=[party_input_audio, party_source_language, party_target_language],
                            outputs=[party_output_audio, party_output_text],
                        )        

            ## CHATBOT ##
            with gr.Accordion("AI Chatbot", open=True):
                with gr.Accordion("Assistant Mode", open=False):
                    with gr.Row():
                        with gr.Column():
                            expert_mode=gr.Dropdown(label="Domain Expert:",choices=chatprompt.domain_experts.keys(), interactive=True)
                        with gr.Column():
                            speech_style=gr.Dropdown(label="Speech and Writing Style:",choices=chatprompt.speech_styles.keys(), interactive=True)                        

                with gr.Accordion("Session Logs", open=False):
                    with gr.Row():
                        chat_logs=gr.Dropdown(label="Log files",choices=filedata.list_session_files(), interactive=True)
                    with gr.Row():
                        btn_load_chat_log=gr.Button(value="Load to chat",size="sm",scale=2,min_width=30,interactive=True)
                        btn_delete_chat_log=gr.Button(value="Delete",size="sm",scale=1,min_width=30,interactive=True)

                chatbot = gr.Chatbot(
                    [],
                    label="Chatbot",
                    elem_id="chatbot",
                    bubble_full_width=False,
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.HTML()
                    with gr.Column(scale=2):
                        server_status_code = gr.HTML(label="api finish status: ")                

                with gr.Row():
                    box_message = gr.Textbox(
                        scale=5,
                        show_label=False,
                        placeholder="Enter text and press enter",
                        container=False,
                    )
                with gr.Row():                    
                    btn_submit = gr.Button("Send",scale=3)
                    btn_clear = gr.ClearButton(
                        [
                            speech_style,
                            expert_mode,
                            box_message,
                            chatbot,
                            upload_youtube_url,
                            upload_web_url,
                            upload_web_search,
                            upload_image_box,
                            upload_audio_box,
                            upload_video_box,
                            upload_file_box,
                            state
                        ],value="Clear & New",size="sm",
                        scale=1
                    )
                    btn_save_chat=gr.Button(value="Save Session",size="sm",scale=1,min_width=30,interactive=True)                        

                with gr.Accordion("Model Parameters", open=False):
                    # server_status_code = gr.Textbox(label="Status code from server")
                    with gr.Row():
                        with gr.Column():
                            api_host = gr.Textbox(label="API base", value="http://192.168.0.29:8004")                                
                        with gr.Column():
                            active_model = gr.Textbox(label="Active model", value="llava:7b-v1.6-mistral-q5_0", info = "current model",)                                
                        with gr.Column():
                            api_key = gr.Textbox(label="API Key", value="ollama", type="password", placeholder="sk..", info = "You have to provide your own GPT4 keys for this app to function properly",)                            
                        with gr.Column():
                            model_dropdown = gr.Dropdown(label="Available models", value="zephyr:latest", choices=chatopenai.list_models(),info = "select a model", interactive=True)

                    ## in-line update event handler
                    def update_active_model(new_model):
                        gr.Info(f"switch model to {new_model}")
                        return gr.Textbox(value=new_model, interactive=True)
                    model_dropdown.change(fn=update_active_model, inputs=[model_dropdown], outputs=[active_model])

                    system_msg_info = """System message helps set the behavior of the AI Assistant. For example, the assistant could be instructed with 'You are a helpful assistant.'"""
                    system_prompt = gr.Textbox(
                        label="Instruct the AI Assistant to set its beaviour",
                        info=system_msg_info,
                        value="You are helpful AI assistant on helping answer user question and research.",
                        placeholder="Type here..",
                        lines=2
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
                            stop_words = gr.Textbox(label="stop words", value="<")
                        with gr.Column():
                            frequency_penalty = gr.Number(
                                label="frequency_penalty", value=0, precision=0
                            )
                        with gr.Column():
                            chat_counter = gr.Number(
                                value=0, visible=False, precision=0
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
                    return filedata.save_text_file("notepad.txt",text_notes)
                def load_notespad(filename:str="notepad.txt"):
                    gr.Info("load notes")                    
                    return filedata.get_text_file(filename)

                btn_save_notepad.click(fn=save_notespad, inputs=[box_notepad])
                btn_load_notepad.click(fn=load_notespad, outputs=[box_notepad])                
            ## update expert mode and speech style
            ## in-line update event handler
            def update_expert_model(expert_mode=None,system_prompt=None, speech_style=None):
                system_prompt = chatprompt.domain_experts[expert_mode]
                gr.Info(f"set expert mode {expert_mode}\n{system_prompt}")                    
                if speech_style is not None and len(speech_style)>0:
                    system_prompt=system_prompt+f"\n You response style should be in {speech_style} style."
                return gr.Textbox(value=system_prompt, interactive=True)
            expert_mode.change(fn=update_expert_model, inputs=[expert_mode,system_prompt,speech_style], outputs=[system_prompt])

            def update_speech_style(expert_mode=None,system_prompt=None, speech_style=None):
                gr.Info(f"switch speech_style to {speech_style}")
                if expert_mode is not None and len(expert_mode)>0:
                    system_prompt = chatprompt.domain_experts[expert_mode]
                if system_prompt is None:
                    system_prompt=f"\nYou response should be in {speech_style} style."                    
                if speech_style is not None and len(speech_style)>0:
                    system_prompt=system_prompt+f"\nYou response should be in {speech_style} style."                    
                return gr.Textbox(value=system_prompt, interactive=True)
            speech_style.change(fn=update_speech_style, inputs=[expert_mode,system_prompt,speech_style], outputs=[system_prompt])

            ## input message
            txt_message = box_message.submit(
                chatopenai.message_and_history_v2,
                inputs=[
                    api_host,api_key,active_model,
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
                chatopenai.message_and_history_v2,
                inputs=[
                    api_host,api_key,active_model,
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
            def load_chat_session(chatlogfile,chatbot,state):
                if chatlogfile is None:
                    gr.Warning(f"no session log selected")            
                    return         
                chatbot,state=filedata.load_session_files(chatlogfile,chatbot,state)
                gr.Info(f"loaded session log {chatlogfile}")
                return chatbot,state

            def save_chat_session(chatbot,state):
                save_chatbot_file=filedata.save_session_files(chatbot,state)
                outputs = filedata.list_session_files()
                gr.Info(f"saved session {save_chatbot_file}")
                return outputs
            
            def delete_chat_session(chatlogfile):
                if chatlogfile is None:
                    gr.Warning(f"no session log selected")            
                    return 
                filedata.delete_session_files(chatlogfile)
                outputs = filedata.list_session_files()
                gr.Info(f"deleted session log {chatlogfile}")
                return outputs

            btn_save_chat.click(fn=save_chat_session,
                                inputs=[chatbot,state],outputs=[chat_logs])
            btn_delete_chat_log.click(fn=delete_chat_session,
                                    inputs=[chat_logs],outputs=[chat_logs])                        
            btn_load_chat_log.click(fn=load_chat_session,
                                    inputs=[chat_logs,chatbot,state],
                                    outputs=[chatbot,state]
                                    )

        ## upload image ##
        upload_image_box.upload(fn=chatopenai.upload_image, 
                                inputs=[
                                    api_host,api_key,active_model,
                                    box_message,upload_image_box,
                                    chatbot,state,system_prompt],
                                outputs=[chatbot, state])

        def transcribe_audio(audio_file,chatbot:list,history:list,model_type:str="distil_whisper"):
            gr.Info("transcribing audio, please wait!")
            if model_type=="distil_whisper":
                chatbot, history = mediadata.distil_whisper(filepath=audio_file,chatbot=chatbot,history=history)
            elif model_type=="faster_whisper":
                chatbot, history = mediadata.faster_whisper(filepath=audio_file,chatbot=chatbot,history=history)
            else:
                raise ValueError(f"unsupported model type {model_type}")
            
            if ckb_save_result.value==True:
                historydata.add_collection_record(colname="upload_audio_collection",weburl=audio_file,filecontent=chatbot[-1][1])

            return chatbot, history

        def transcribe_video(audio_file,chatbot:list,history:list,model_type:str="distil_whisper"):
            gr.Info("transcribing video, please wait!")
            chatbot, history = mediadata.transcribe_video(filepath=audio_file,chatbot=chatbot,history=history,
                                                export_transcriber=model_type)
            if ckb_save_result.value==True:
                historydata.add_collection_record(colname="upload_video_collection",weburl=audio_file,filecontent=chatbot[-1][1])
            
            return chatbot, history
        
        def transcribe_youtube(fileurl,chatbot:list,history:list,model_type:str="distil_whisper"):
            gr.Info("transcribing youtube video, please wait!")
            chatbot, history = mediadata.transcribe_youtube(input_url=fileurl,chatbot=chatbot,history=history,
                                                export_transcriber=model_type)
            if ckb_save_result.value==True:
                historydata.add_collection_record(colname="youtube_video_collection",weburl=fileurl,filecontent=chatbot[-1][1])
            
            return chatbot, history

        def scrape_web(website_url,chatbot:list,history:list,tool:str="newspaper"):
            gr.Info("scrapping web page, please wait!")
            chatbot, history = webdata.scrap_web(website_url=website_url,chatbot=chatbot,history=history,
                                                tool=tool)
            if ckb_save_result.value==True:
                historydata.add_collection_record(colname="web_scrapping_collection",weburl=website_url,filecontent=chatbot[-1][1])
            
            return chatbot, history

        def web_search(keywords,chatbot:list,history:list,tool:str="duckduckgo"):
            gr.Info("searching web, please wait!")
            chatbot, history = webdata.web_search(keywords=keywords,chatbot=chatbot,history=history,tool=tool)
            if ckb_save_result.value==True:
                historydata.add_collection_record(colname="web_search_collection",weburl=keywords,filecontent=chatbot[-1][1])
            return chatbot, history

        upload_audio_box.upload(fn=transcribe_audio,
                                inputs=[upload_audio_box,chatbot,state],
                                outputs=[chatbot, state] 
                                )
        
        upload_video_box.upload(fn=transcribe_video,
                                inputs=[upload_video_box,chatbot,state],
                                outputs=[chatbot, state] 
                                )        
        upload_file_box.upload(fn=filedata.load_file_content,
                                inputs=[upload_file_box,chatbot,state],
                                outputs=[chatbot, state]                                
                               )

        upload_youtube_url.submit(fn=transcribe_youtube, 
                                inputs=[upload_youtube_url,chatbot,state],
                                outputs=[chatbot, state]                                   
                                )
        upload_web_url.submit(fn=scrape_web, 
                                inputs=[upload_web_url,chatbot,state,web_scrapping_tool],
                                outputs=[chatbot, state]                                   
                                )        

        upload_web_search.submit(fn=web_search,
                                inputs=[upload_web_search,chatbot,state,web_search_tool],
                                outputs=[chatbot, state]                                                                                      
                             )

        history_document_id.submit(fn=historydata.get_document,
                                        inputs=[history_cols,history_document_id,chatbot,state], 
                                        outputs=[chatbot,state]
                                        )

if __name__ == "__main__":
    demoui.queue()
    demoui.launch(debug=True)
