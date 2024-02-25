
import gradio as gr
import pandas as pd
import os
import time
import filedata 
import websearch
import webdata

class WebResearchCrew:
    def build_webresearch_ui(self):
        self.blocks_websearch = gr.Blocks(analytics_enabled=False)
        with self.blocks_websearch:
            with gr.Accordion("Latest News Search", open=True):
                gr.Markdown("get latest Yahoo news in headlines and full detail format.")
                box_on_demand_search = gr.Textbox(
                    lines=1,
                    info="write one line of search text here then please Enter or click search button",
                    label="Search Input",
                    interactive=True,
                    autofocus=True
                )
                box_on_demand_Result = gr.DataFrame(label="Search Result",
                                                    wrap=True,
                                                    interactive=False)  
                md_on_demand_full_result=gr.Markdown()            
                with gr.Row():
                    with gr.Column(scale=3):
                        btn_on_demand_search = gr.Button(
                            size="sm", value="Web Search"
                        )
                    with gr.Column(scale=1):
                        btn_on_demand_clear = gr.ClearButton(components=[md_on_demand_full_result,box_on_demand_Result,box_on_demand_search,btn_on_demand_search])

                ## function: scratch pad
                def on_demand_search(query:str):
                    gr.Info("performing search, please wait!")
                    news_data,news_data_df = websearch.SearchNewsDB.latest_yahoo_news_content(topic=query)
                    #web_search_result,web_search_result2 = webdata.web_search(keywords=query)
                    #return f"<h2>Yahoo Latest News</h2><p>{latest_yahoo_news}</p><hr/>"
                    return news_data_df,news_data
                    #return f"<h2>Yahoo Latest News</h2><p>{latest_yahoo_news}</p><hr/><h2>Web Search Result</h2><p>{web_search_result}</p>"
                
                box_on_demand_search.submit(fn=on_demand_search, inputs=[box_on_demand_search], outputs=[box_on_demand_Result,md_on_demand_full_result])
                btn_on_demand_search.click(fn=on_demand_search, inputs=[box_on_demand_search], outputs=[box_on_demand_Result,md_on_demand_full_result])

                ## update expert mode and speech style
            with gr.Accordion("Web Search Summary (Experimental)", open=False):
                gr.Markdown("generate easy to digest result summary report. Limitation: Duckduckgo has usage limit per day")
                box_report_search = gr.Textbox(
                    lines=1,
                    info="write a search topic.",
                    label="Search Input",
                    interactive=True,
                    autofocus=True
                )
                box_report_search_result = gr.Markdown()
                with gr.Row():
                    with gr.Column(scale=3):
                        btn_report_search = gr.Button(size="sm", value="Create Search Report")
                    with gr.Column(scale=1):
                        btn_report_clear = gr.ClearButton(components=[box_report_search,box_report_search_result])

                ## function
                def search_report(query):
                    gr.Warning("creating search report, please wait! may take few minutes to complete.")
                    websearch.SearchNewsDB.preparenews(query=query)
                    report = websearch.SearchNewsDB.searchreport(query=query)
                    return f"<h2>Search Report</h2>{report}"

                box_report_search.submit(fn=search_report, inputs=[box_report_search], outputs=[box_report_search_result])
                btn_report_search.click(fn=search_report, inputs=[box_report_search], outputs=[box_report_search_result])
                ## update expert mode and speech style
            with gr.Accordion("Web Search by Crew AI Agents (Experimental)", open=False):
                gr.Markdown("Assign task to a crew of AI agents to perform web search and write a summary. Limitation: Duckduckgo has usage limit per day")
                box_newscrew_search = gr.Textbox(
                    lines=1,
                    info="write search task for the crew on news searcher and writer.",
                    interactive=True,
                    label="Search Input",
                    autofocus=True
                )
                box_newscrew_search_result=gr.Markdown()
                with gr.Row():
                    with gr.Column(scale=3):
                        btn_newscrew = gr.Button(size="sm", value="Submit Crew Search Task")
                    with gr.Column(scale=1):
                        btn_newscrew_clear = gr.ClearButton(components=[box_newscrew_search,box_newscrew_search_result])

                ## function
                def submit_crew_search_task(query):
                    gr.Warning("process crew search task submission, please wait... may take few minutes to complete.")
                    crew_report = websearch.NewsCrew.run_search_job(query=query)
                    return f"<h2>Crew Report</h2>{crew_report}"

                box_newscrew_search.submit(fn=submit_crew_search_task, inputs=[box_newscrew_search],outputs=[box_newscrew_search_result])
                btn_newscrew.click(fn=submit_crew_search_task, inputs=[box_newscrew_search],outputs=[box_newscrew_search_result])

        return self.blocks_websearch
    
class AppMain(WebResearchCrew):

    def __init__(self):
        super().__init__()

    def main(self):
        self.build_webresearch_ui()
        self.app = gr.TabbedInterface(
            [self.blocks_websearch],
            ["Web Search"],
            title="PavAI Productivity Workspace",
            analytics_enabled=False
        )

    def launch(self):
        self.main()
        self.app.queue().launch()

if __name__ == "__main__":
    app = AppMain()
    app.launch()
