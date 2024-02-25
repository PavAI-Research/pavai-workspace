import gradio as gr
import pandas as pd
import os
import time
import filedata 
import multilingual 
import lang_list 

class CommunicationTranslator:
    def build_translator_ui(self):
        self.blocks_translator = gr.Blocks(analytics_enabled=False)
        with self.blocks_translator:
            state = gr.State()
            with gr.Row():
                with gr.Column(scale=7):
                    ## Seamless Communication ##
                    with gr.Accordion("Speech-to-Speech Translator", open=True):
                        gr.Markdown("Speak on your own native language to another party using realtime speech-to-speech translation.")
                        with gr.Group(visible=True) as voicechat:
                            with gr.Row():
                                ##  USER INPUT
                                with gr.Column(scale=1):
                                    source_language = gr.Dropdown(
                                        label="User Spoken language",
                                        choices=lang_list.ASR_TARGET_LANGUAGE_NAMES,
                                        value="English",
                                    )
                                    input_audio = gr.Audio(
                                        sources=["microphone", "upload"],
                                        label="User voice",
                                        type="filepath",
                                        scale=1,
                                        min_width=20,
                                    )
                                    # user_speech_text = gr.Text(label="Your voice text", lines=2)
                                    btn_translate = gr.Button(
                                        "Translate Your Speech to Party Language",
                                        size="sm",
                                        scale=1,
                                    )  
                                ## USER TARGET
                                with gr.Column(scale=1):
                                    with gr.Group():                                        
                                        target_language = gr.Dropdown(
                                            label="Target Party language",
                                            choices=lang_list.S2ST_TARGET_LANGUAGE_NAMES,
                                            value=multilingual.DEFAULT_TARGET_LANGUAGE,
                                        )
                                        xspeaker = gr.Slider(
                                            1,
                                            100,
                                            value=7,
                                            step=1,
                                            label="Speaker Id",
                                            interactive=True,
                                        )
                                        output_audio = gr.Audio(
                                            label="Translated speech",
                                            autoplay=True,
                                            streaming=False,
                                            type="numpy",
                                        )
                                        output_text = gr.Textbox(
                                            label="Translated text", lines=3
                                        )
                                        btn_clear = gr.ClearButton(
                                            [
                                                source_language,
                                                input_audio,
                                                output_audio,
                                                output_text,
                                            ],
                                            size="sm",
                                        )
                            # with gr.Row():
                            #     gr.HTML("<hr size=1/>")
                            with gr.Row():

                                ## PARTY INPUT
                                with gr.Column(scale=1):
                                    party_source_language = gr.Dropdown(
                                        label="Party Spoken language",
                                        choices=lang_list.ASR_TARGET_LANGUAGE_NAMES,
                                        value="French",
                                    )
                                    party_input_audio = gr.Audio(
                                        sources=["microphone", "upload"],
                                        label="Party voice",
                                        type="filepath",
                                        scale=1,
                                        min_width=20,
                                    )
                                    # party_speech_text = gr.Text(label="Party voice text", lines=2)
                                    party_btn_translate = gr.Button(
                                        "Translate Party Speech to Your Language",
                                        size="sm",
                                        scale=1,
                                    )

                                ## Other Party
                                with gr.Column(scale=1):
                                    with gr.Group():
                                        party_target_language = gr.Dropdown(
                                            label="Target User language",
                                            choices=lang_list.S2ST_TARGET_LANGUAGE_NAMES,
                                            value="English",
                                        )
                                        party_xspeaker = gr.Slider(
                                            1,
                                            100,
                                            value=7,
                                            step=1,
                                            label="Speaker Id",
                                            interactive=True,
                                        )
                                        party_output_audio = gr.Audio(
                                            label="Translated speech",
                                            autoplay=True,
                                            streaming=False,
                                            type="numpy",
                                        )
                                        party_output_text = gr.Textbox(
                                            label="Translated text", lines=3
                                        )
                                        party_btn_clear = gr.ClearButton(
                                            [
                                                party_source_language,
                                                party_input_audio,
                                                party_output_audio,
                                                party_output_text,
                                            ],
                                            size="sm",
                                        )

                                        # handle speaker id change
                                        party_xspeaker.change(
                                            fn=multilingual.SeamlessM4T().update_value, inputs=xspeaker
                                        )
                                # handle
                                btn_translate.click(
                                    fn=multilingual.SeamlessM4T().run_s2st,
                                    inputs=[
                                        input_audio,
                                        source_language,
                                        target_language,
                                        xspeaker,
                                    ],
                                    outputs=[output_audio, output_text],
                                    api_name="s2st",
                                )

                                # handle
                                party_btn_translate.click(
                                    fn=multilingual.SeamlessM4T().run_s2st,
                                    inputs=[
                                        party_input_audio,
                                        party_source_language,
                                        party_target_language,
                                        party_xspeaker,
                                    ],
                                    outputs=[party_output_audio, party_output_text],
                                    api_name="s2st_party",
                                )

                                ## auto submit
                                input_audio.stop_recording(
                                    fn=multilingual.SeamlessM4T().run_s2st,
                                    inputs=[
                                        input_audio,
                                        source_language,
                                        target_language,
                                    ],
                                    outputs=[output_audio, output_text],
                                )

                                ## auto submit
                                party_input_audio.stop_recording(
                                    fn=multilingual.SeamlessM4T().run_s2st,
                                    inputs=[
                                        party_input_audio,
                                        party_source_language,
                                        party_target_language,
                                    ],
                                    outputs=[party_output_audio, party_output_text],
                                )

            with gr.Row():
                gr.Markdown("Limitations: if you have a microphoe plugin but not seen in the browser. the issue could be the url is not localhost or secure connection. please consult the browser settings.")                    
        return self.blocks_translator

# # theme=gr.themes.Monochrome()
#theme=gr.themes.Glass())

class AppMain(CommunicationTranslator):

    def __init__(self):
        super().__init__()

    def main(self):
        self.build_translator_ui()
        self.app = gr.TabbedInterface(
            [self.blocks_translator],
            ["Seamless Communication"],
            title="PavAI Productivity Workspace",
            analytics_enabled=False
        )

    def launch(self):
        self.main()
        self.app.queue().launch()


if __name__ == "__main__":
    app = AppMain()
    app.launch()
