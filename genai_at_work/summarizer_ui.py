from genai_at_work import config, logutil
logger = logutil.logging.getLogger(__name__)

import gradio as gr
import pandas as pd
import os
import time
import filedata 
import summarizer
import mediadata

translation_lang_names=['Afrikaans', 'Akan', 'Albanian', 'Amharic', 'Arabic', 'Armenian', 'Assamese', 'Aymara', 'Azerbaijani', 'Bangla', 'Basque', 'Belarusian', 'Bhojpuri', 'Bosnian', 'Bulgarian', 'Burmese', 'Catalan', 'Cebuano', 'Chinese (Simplified)', 'Chinese (Traditional)', 'Corsican', 'Croatian', 'Czech', 'Danish', 'Divehi', 'Dutch', 'English', 'Esperanto', 'Estonian', 'Ewe', 'Filipino', 'Finnish', 'French', 'Galician', 'Ganda', 'Georgian', 'German', 'Greek', 'Guarani', 'Gujarati', 'Haitian Creole', 'Hausa', 'Hawaiian', 'Hebrew', 'Hindi', 'Hmong', 'Hungarian', 'Icelandic', 'Igbo', 'Indonesian', 'Irish', 'Italian', 'Japanese', 'Javanese', 'Kannada', 'Kazakh', 'Khmer', 'Kinyarwanda', 'Korean', 'Krio', 'Kurdish', 'Kyrgyz', 'Lao', 'Latin', 'Latvian', 'Lingala', 'Lithuanian', 'Luxembourgish', 'Macedonian', 'Malagasy', 'Malay', 'Malayalam', 'Maltese', 'Māori', 'Marathi', 'Mongolian', 'Nepali', 'Northern Sotho', 'Norwegian', 'Nyanja', 'Odia', 'Oromo', 'Pashto', 'Persian', 'Polish', 'Portuguese', 'Punjabi', 'Quechua', 'Romanian', 'Russian', 'Samoan', 'Sanskrit', 'Scottish Gaelic', 'Serbian', 'Shona', 'Sindhi', 'Sinhala', 'Slovak', 'Slovenian', 'Somali', 'Southern Sotho', 'Spanish', 'Sundanese', 'Swahili', 'Swedish', 'Tajik', 'Tamil', 'Tatar', 'Telugu', 'Thai', 'Tigrinya', 'Tsonga', 'Turkish', 'Turkmen', 'Ukrainian', 'Urdu', 'Uyghur', 'Uzbek', 'Vietnamese', 'Welsh', 'Western Frisian', 'Xhosa', 'Yiddish', 'Yoruba', 'Zulu']
translation_lang_codes=['af', 'ak', 'sq', 'am', 'ar', 'hy', 'as', 'ay', 'az', 'bn', 'eu', 'be', 'bho', 'bs', 'bg', 'my', 'ca', 'ceb', 'zh-Hans', 'zh-Hant', 'co', 'hr', 'cs', 'da', 'dv', 'nl', 'en', 'eo', 'et', 'ee', 'fil', 'fi', 'fr', 'gl', 'lg', 'ka', 'de', 'el', 'gn', 'gu', 'ht', 'ha', 'haw', 'iw', 'hi', 'hmn', 'hu', 'is', 'ig', 'id', 'ga', 'it', 'ja', 'jv', 'kn', 'kk', 'km', 'rw', 'ko', 'kri', 'ku', 'ky', 'lo', 'la', 'lv', 'ln', 'lt', 'lb', 'mk', 'mg', 'ms', 'ml', 'mt', 'mi', 'mr', 'mn', 'ne', 'nso', 'no', 'ny', 'or', 'om', 'ps', 'fa', 'pl', 'pt', 'pa', 'qu', 'ro', 'ru', 'sm', 'sa', 'gd', 'sr', 'sn', 'sd', 'si', 'sk', 'sl', 'so', 'st', 'es', 'su', 'sw', 'sv', 'tg', 'ta', 'tt', 'te', 'th', 'ti', 'ts', 'tr', 'tk', 'uk', 'ur', 'ug', 'uz', 'vi', 'cy', 'fy', 'xh', 'yi', 'yo', 'zu']
translation_langs={'Afrikaans': 'af', 'Akan': 'ak', 'Albanian': 'sq', 'Amharic': 'am', 'Arabic': 'ar', 'Armenian': 'hy', 'Assamese': 'as', 'Aymara': 'ay', 'Azerbaijani': 'az', 'Bangla': 'bn', 'Basque': 'eu', 'Belarusian': 'be', 'Bhojpuri': 'bho', 'Bosnian': 'bs', 'Bulgarian': 'bg', 'Burmese': 'my', 'Catalan': 'ca', 'Cebuano': 'ceb', 'Chinese (Simplified)': 'zh-Hans', 'Chinese (Traditional)': 'zh-Hant', 'Corsican': 'co', 'Croatian': 'hr', 'Czech': 'cs', 'Danish': 'da', 'Divehi': 'dv', 'Dutch': 'nl', 'English': 'en', 'Esperanto': 'eo', 'Estonian': 'et', 'Ewe': 'ee', 'Filipino': 'fil', 'Finnish': 'fi', 'French': 'fr', 'Galician': 'gl', 'Ganda': 'lg', 'Georgian': 'ka', 'German': 'de', 'Greek': 'el', 'Guarani': 'gn', 'Gujarati': 'gu', 'Haitian Creole': 'ht', 'Hausa': 'ha', 'Hawaiian': 'haw', 'Hebrew': 'iw', 'Hindi': 'hi', 'Hmong': 'hmn', 'Hungarian': 'hu', 'Icelandic': 'is', 'Igbo': 'ig', 'Indonesian': 'id', 'Irish': 'ga', 'Italian': 'it', 'Japanese': 'ja', 'Javanese': 'jv', 'Kannada': 'kn', 'Kazakh': 'kk', 'Khmer': 'km', 'Kinyarwanda': 'rw', 'Korean': 'ko', 'Krio': 'kri', 'Kurdish': 'ku', 'Kyrgyz': 'ky', 'Lao': 'lo', 'Latin': 'la', 'Latvian': 'lv', 'Lingala': 'ln', 'Lithuanian': 'lt', 'Luxembourgish': 'lb', 'Macedonian': 'mk', 'Malagasy': 'mg', 'Malay': 'ms', 'Malayalam': 'ml', 'Maltese': 'mt', 'Māori': 'mi', 'Marathi': 'mr', 'Mongolian': 'mn', 'Nepali': 'ne', 'Northern Sotho': 'nso', 'Norwegian': 'no', 'Nyanja': 'ny', 'Odia': 'or', 'Oromo': 'om', 'Pashto': 'ps', 'Persian': 'fa', 'Polish': 'pl', 'Portuguese': 'pt', 'Punjabi': 'pa', 'Quechua': 'qu', 'Romanian': 'ro', 'Russian': 'ru', 'Samoan': 'sm', 'Sanskrit': 'sa', 'Scottish Gaelic': 'gd', 'Serbian': 'sr', 'Shona': 'sn', 'Sindhi': 'sd', 'Sinhala': 'si', 'Slovak': 'sk', 'Slovenian': 'sl', 'Somali': 'so', 'Southern Sotho': 'st', 'Spanish': 'es', 'Sundanese': 'su', 'Swahili': 'sw', 'Swedish': 'sv', 'Tajik': 'tg', 'Tamil': 'ta', 'Tatar': 'tt', 'Telugu': 'te', 'Thai': 'th', 'Tigrinya': 'ti', 'Tsonga': 'ts', 'Turkish': 'tr', 'Turkmen': 'tk', 'Ukrainian': 'uk', 'Urdu': 'ur', 'Uyghur': 'ug', 'Uzbek': 'uz', 'Vietnamese': 'vi', 'Welsh': 'cy', 'Western Frisian': 'fy', 'Xhosa': 'xh', 'Yiddish': 'yi', 'Yoruba': 'yo', 'Zulu': 'zu'}

class DocumentSummarizer:
    def build_summarizer_ui(self):
        self.blocks_summarizer = gr.Blocks(analytics_enabled=False)
        with self.blocks_summarizer:
            gr.Markdown("#### Extract and generate high quality summary of large pdf/text file or long text.")
            with gr.Accordion("Summarize File", open=False):
                slide_pages_limit = gr.Slider(minimum=1, maximum=1600, value=38,label="maximum pages", interactive=True)
                summary_upload_file = gr.File(label="Upload file",file_types=["txt,pdf"])
                with gr.Row():
                    btn_summarize_file = gr.Button(size="sm", value="summarize file", interactive=True)                
                with gr.Row():                    
                    box_file_summary_text = gr.TextArea(
                        lines=7,
                        info="result summary text.",
                        interactive=True,
                        text_align="left",
                        show_copy_button=True,
                    )

                def generate_summary_files(filename: str, max_pages:int=138):
                    gr.Info("generate file summary, please wait!")
                    return summarizer.GeneralTextSummarizer().summarize_filetype(filename=filename, max_pages=max_pages)

                btn_summarize_file.click(fn=generate_summary_files, inputs=[summary_upload_file,slide_pages_limit], outputs=[box_file_summary_text])

            with gr.Accordion("Summarize Text", open=False):
                box_input_text = gr.TextArea(
                    lines=5,
                    info="Paste or write text for summarization",
                    interactive=True,
                    text_align="left",                    
                    show_copy_button=True,
                    autoscroll=True,      
                    autofocus=True,
                    label="Input Text"              
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        btn_summarize_input_text = gr.Button(size="sm", value="generate text summarization", interactive=True)
                with gr.Row():
                    box_summarized_text = gr.TextArea(
                        lines=3,
                        info="result summary",
                        interactive=True,
                        text_align="left",
                        show_copy_button=True,
                        autoscroll=True, 
                        label="Output Summary"                                           
                    )         
                with gr.Row():                               
                    with gr.Column(scale=1):                        
                        btn_summarize_input_clear = gr.ClearButton(components=[box_input_text,box_summarized_text])                                        

                def generate_summary_input_text(text_notes):
                    gr.Info("generate input text summary, please wait!")
                    return summarizer.GeneralTextSummarizer().summarize_text(long_text=text_notes)
                
                btn_summarize_input_text.click(fn=generate_summary_input_text, inputs=[box_input_text], outputs=[box_summarized_text])

            with gr.Accordion("Summarize Youtube Video Transcription", open=False):
                box_input_youtube_url = gr.Textbox(
                    lines=1,
                    info="enter youtube video-id here",
                    interactive=True,
                    text_align="left",                    
                    show_copy_button=True,
                    autoscroll=True,      
                    autofocus=True,
                    label="Youtube Video Id"              
                )
                with gr.Row():
                    btn_summarize_youtube_url = gr.Button(size="sm", value="Summarize transcription", interactive=True)
                with gr.Row():
                    box_youtube_transcribed_text = gr.TextArea(
                        lines=5,
                        info="Input youtube video Id",
                        interactive=True,
                        text_align="left",
                        show_copy_button=True,
                        autoscroll=True, 
                        label="Transcribed youtube video text"                                           
                    )      
                with gr.Row():                                                   
                    box_youtube_summarized_text = gr.TextArea(
                        lines=3,
                        info="Result",
                        interactive=True,
                        text_align="left",
                        show_copy_button=True,
                        autoscroll=True, 
                        label="Output Summary"                                           
                    )      
                with gr.Row():                                                   
                    btn_summarize_input_clear = gr.ClearButton(components=[box_youtube_transcribed_text,box_youtube_summarized_text])                                        

                def generate_summary_youtube_video(video_id):
                    gr.Info("getting youtube video transcript, please wait!")
                    script_text = mediadata.download_youtube_transcript(video_id=video_id.strip())
                    gr.Info("summarizing, please wait!")                            
                    result = summarizer.GeneralTextSummarizer().summarize_text(long_text=script_text)                    
                    return script_text, result
                
                btn_summarize_youtube_url.click(fn=generate_summary_youtube_video, 
                                                       inputs=[box_input_youtube_url], 
                                                       outputs=[box_youtube_transcribed_text,
                                                                box_youtube_summarized_text])

            # with gr.Accordion("Translate Youtube Video", open=False):
            #     with gr.Row():
            #         with gr.Column(scale=1):                                         
            #             box_tx_youtube_url = gr.Textbox(
            #             lines=1,
            #             info="enter youtube video-id here",
            #             interactive=True,
            #             text_align="left",                    
            #             show_copy_button=True,
            #             autoscroll=True,      
            #             autofocus=True,
            #             label="1. Enter Youtube Video Id"              
            #         )
            #         with gr.Column(scale=1):                    
            #             box_youtube_tx_lang = gr.Dropdown(choices=translation_lang_names, interactive=True,label="2. Select Target Languages", value="en")                        
            #     with gr.Row():
            #         btn_youtube_get_tx = gr.Button(size="sm", value="3. Get translation and Summary", interactive=True)
            #     with gr.Row():
            #         with gr.Column(scale=1):                    
            #             box_youtube_tx_text = gr.TextArea(
            #                 lines=5,
            #                 info="Input youtube video Id",
            #                 interactive=True,
            #                 text_align="left",
            #                 show_copy_button=True,
            #                 autoscroll=True, 
            #                 label="Transcribed youtube video text"                                           
            #             )      

            #     with gr.Row():                                                   
            #         box_youtube_summarized_tx_text = gr.TextArea(
            #             lines=3,
            #             info="Result",
            #             interactive=True,
            #             text_align="left",
            #             show_copy_button=True,
            #             autoscroll=True, 
            #             label="Output Summary"                                           
            #         )      
            #     with gr.Row():                                                   
            #         btn_tx_input_clear = gr.ClearButton(components=[box_youtube_tx_text,box_youtube_summarized_tx_text])                                        

            #     def translate_summary_youtube_video(video_id,target_language):
            #         gr.Info("getting youtube video transcript, please wait!")
            #         target_language_code=translation_langs[target_language]
            #         script_text = mediadata.download_youtube_translate(video_id=video_id.strip(),languages=[target_language_code])
            #         gr.Info("summarizing, please wait!")                            
            #         result = summarizer.GeneralTextSummarizer().summarize_text(long_text=script_text)                    
            #         return script_text, result
                
            #     btn_youtube_get_tx.click(fn=translate_summary_youtube_video, 
            #                                            inputs=[box_tx_youtube_url,box_youtube_tx_lang], 
            #                                            outputs=[box_youtube_tx_text,
            #                                                     box_youtube_summarized_tx_text])


        return self.blocks_summarizer
    
class AppMain(DocumentSummarizer):

    def __init__(self):
        super().__init__()

    def main(self):
        self.build_summarizer_ui()
        self.app = gr.TabbedInterface(
            [self.blocks_summarizer],
            ["Long-Text Summarizer"],
            title="PavAI Productivity Workspace",
            analytics_enabled=False
        )

    def launch(self):
        self.main()
        self.app.queue().launch()

if __name__ == "__main__":
    app = AppMain()
    app.launch()
