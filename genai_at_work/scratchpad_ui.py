from genai_at_work import config, logutil
logger = logutil.logging.getLogger(__name__)

import gradio as gr
import pandas as pd
import os
import time
import filedata 

class ScratchPad:
    def build_scratchpad_ui(self):
        self.blocks_scratchpad = gr.Blocks(analytics_enabled=False)
        with self.blocks_scratchpad:
            with gr.Accordion("Important tasks to complete", open=True):
                box_notepad = gr.TextArea(
                    lines=3,
                    info="write single page tasks.",
                    interactive=True,
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        btn_save_notepad = gr.Button(
                            size="sm", value="save and update"
                        )
                    with gr.Column(scale=1):
                        btn_load_notepad = gr.Button(size="sm", value="load")

                ## function: scratch pad
                def save_notespad(text_notes):
                    gr.Info("saved notes to tasks.txt!")
                    return filedata.save_text_file("workspace/scratchpad/tasks.txt", text_notes)

                def load_notespad(filename: str = "workspace/scratchpad/tasks.txt"):
                    gr.Info("load notes")
                    return filedata.get_text_file(filename)

                btn_save_notepad.click(fn=save_notespad, inputs=[box_notepad])
                btn_load_notepad.click(fn=load_notespad, outputs=[box_notepad])
                ## update expert mode and speech style
            with gr.Accordion("Notes", open=False):
                box_notepad = gr.TextArea(
                    lines=5,
                    info="write single page quick notes.",
                    interactive=True,
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        btn_save_notepad = gr.Button(
                            size="sm", value="save and update"
                        )
                    with gr.Column(scale=1):
                        btn_load_notepad = gr.Button(size="sm", value="load")

                ## function: scratch pad
                def save_notespad(text_notes):
                    gr.Info("saved notes to notes.txt!")
                    return filedata.save_text_file("workspace/scratchpad/notes.txt", text_notes)

                def load_notespad(filename: str = "workspace/scratchpad/notes.txt"):
                    gr.Info("load notes")
                    return filedata.get_text_file(filename)

                btn_save_notepad.click(fn=save_notespad, inputs=[box_notepad])
                btn_load_notepad.click(fn=load_notespad, outputs=[box_notepad])
                ## update expert mode and speech style
            with gr.Accordion("Todos", open=False):
                box_notepad = gr.TextArea(
                    lines=3,
                    info="write single page of Todos.",
                    interactive=True,
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        btn_save_notepad = gr.Button(
                            size="sm", value="save and update"
                        )
                    with gr.Column(scale=1):
                        btn_load_notepad = gr.Button(size="sm", value="load")

                ## function: scratch pad
                def save_notespad(text_notes):
                    gr.Info("saved notes to todos.txt!")
                    return filedata.save_text_file("workspace/scratchpad/todos.txt", text_notes)

                def load_notespad(filename: str = "workspace/scratchpad/todos.txt"):
                    gr.Info("load notes")
                    return filedata.get_text_file(filename)

                btn_save_notepad.click(fn=save_notespad, inputs=[box_notepad])
                btn_load_notepad.click(fn=load_notespad, outputs=[box_notepad])
                ## update expert mode and speech style
            with gr.Accordion("Reminders", open=False):
                box_notepad = gr.TextArea(
                    lines=3,
                    info="write single page of Todos.",
                    interactive=True,
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        btn_save_notepad = gr.Button(
                            size="sm", value="save and update"
                        )
                    with gr.Column(scale=1):
                        btn_load_notepad = gr.Button(size="sm", value="load")

                ## function: scratch pad
                def save_notespad(text_notes):
                    gr.Info("saved notes to reminder.txt!")
                    return filedata.save_text_file("workspace/scratchpad/reminder.txt", text_notes)

                def load_notespad(filename: str = "workspace/scratchpad/reminder.txt"):
                    gr.Info("load notes")
                    return filedata.get_text_file(filename)

                btn_save_notepad.click(fn=save_notespad, inputs=[box_notepad])
                btn_load_notepad.click(fn=load_notespad, outputs=[box_notepad])
                ## update expert mode and speech style
        return self.blocks_scratchpad
    