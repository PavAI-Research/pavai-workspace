#!pip install textsum
#!pip install clean-text
#!pip install python-doctr
# tf2onnx
#!pip install pyspellchecker
#!pip install textsum
#!pip install pytorch

import torch
import logging
import pathlib
from datetime import datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
## pdf2text
from cleantext import clean
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from textsum import pdf2text
## textsum
from textsum.summarize import Summarizer
logging.basicConfig(level=logging.ERROR,format="%(asctime)s %(levelname)s %(message)s",datefmt="%m/%d/%Y %I:%M:%S",)
from typing import Any, List, Union, Optional, Sequence, Mapping, Literal, Dict

class SummarizerModelSize(Enum):
    def __str__(self):
        return str(self.value)

    BASE = "pszemraj/led-base-book-summary"
    LARGE = "pszemraj/led-large-book-summary"
    ##LONG = "pszemraj/long-t5-tglobal-base-16384-book-summary"
    ##XL = "pszemraj/long-t5-tglobal-xl-16384-book-summary"

class MetaSingleton(type):
    """
    Metaclass for implementing the Singleton pattern.
    """
    _instances: Dict[type, Any] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super(
                MetaSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Singleton(object, metaclass=MetaSingleton):
    """
    Base class for implementing the Singleton pattern.
    """

    def __init__(self):
        super(Singleton, self).__init__()

class GeneralTextSummarizer(Singleton):
    _summarizer = None
    _ocr_model = None

    def __init__(
        self,
        model_name: str = SummarizerModelSize.BASE,
        use_cuda: bool = True,
        batch_stride: int = 16,
        token_batch_length: int = 2048,
        compile_model: bool = False,
        min_length: int = 8,
        max_length: int = 256,
        kwargs={
            "no_repeat_ngram_size": 3,
            "encoder_no_repeat_ngram_size": 3,
            "repetition_penalty": 2.5,
            "num_beams": 4,
            "num_beam_groups": 1,
            "length_penalty": 0.8,
            "early_stopping": True,
            "do_sample": False,
        },
    ):
        use_cuda = True if torch.cuda.is_available() else False
        kwargs["min_length"] = min_length
        kwargs["max_length"] = max_length
        ## rule
        # batch_stride = int(token_batch_length * max_length_ratio),
        self._summarizer = Summarizer(
            use_cuda=use_cuda,
            model_name_or_path=model_name,  # you can use any Seq2Seq model on the Hub
            batch_stride=batch_stride,
            token_batch_length=token_batch_length,  # tokens to batch summarize at a time, up to 16384
            compile_model=compile_model,
            kwargs=kwargs
        )

    @classmethod
    def model_init(
        cls,
        model_name: str = SummarizerModelSize.BASE,
        use_cuda: bool = True,
        batch_stride: int = 16,
        token_batch_length: int = 2048,
        compile_model: bool = False,
        min_length: int = 8,
        max_length: int = 256,
        kwargs={
            "no_repeat_ngram_size": 3,
            "encoder_no_repeat_ngram_size": 3,
            "repetition_penalty": 2.5,
            "num_beams": 4,
            "num_beam_groups": 1,
            "length_penalty": 0.8,
            "early_stopping": True,
            "do_sample": False,
        },
    ):
        """
        available model sizes:
        - pszemraj/led-base-book-summary
        - pszemraj/led-large-book-summary
        """
        use_cuda = True if torch.cuda.is_available() else False
        kwargs["min_length"] = min_length
        kwargs["max_length"] = max_length
        ## rule
        # batch_stride = int(token_batch_length * max_length_ratio),
        instance = cls()
        instance._summarizer = Summarizer(
            use_cuda=use_cuda,
            model_name_or_path=model_name,  # you can use any Seq2Seq model on the Hub
            batch_stride=batch_stride,
            token_batch_length=token_batch_length,  # tokens to batch summarize at a time, up to 16384
            compile_model=compile_model,
            kwargs=kwargs,
        )
        return instance

    def summarize_text(self, long_text: str) -> str:
        out_str = self._summarizer.summarize_string(long_text)
        return out_str

    def get_ocr_model(self):
        self._ocr_model = ocr_predictor(
            det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True
        )
        return self._ocr_model

    def summarize_text_file(self, filepath: str):
        out_path = self._summarizer.summarize_file(filepath)
        print(f"summary saved to {out_path}")
        return out_path

    def summarize_filetype(
        self,
        filename: str,
        max_pages: int = 138,
        save_ocr_text: bool = False,        
    ) -> str:
        file_path = Path(filename)
        try:
            if file_path.suffix == ".txt":
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    raw_text = f.read()
                text = clean(raw_text, lower=False)
                return self.summarize_text(text)
            elif file_path.suffix == ".pdf":
                logging.info(f"Loading PDF file {file_path}")
                ### ocr_model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
                ocr_model = self.get_ocr_model()
                conversion_stats = pdf2text.convert_PDF_to_Text(
                    file_path,
                    ocr_model=ocr_model,
                    max_pages=max_pages,
                )
                text = conversion_stats["converted_text"]
                if save_ocr_text:
                    pathlib.Path(f"{filename}_ocr.txt").write_text(text)
                return self.summarize_text(text)
            else:
                logging.error(f"Unknown file type {file_path.suffix}")
                text = "ERROR - check example path"
            return text
        except Exception as e:
            logging.info(f"Trying to load file with path {file_path}, error: {e}")
            return "Error: Could not read file. Ensure that it is a valid text file with encoding UTF-8 if text, and a PDF if PDF."

    @staticmethod
    def get_timestamp() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

# class QueryPDF(Singleton):

#     def querypdf(self, query):
#         from gradio_pdf import PDF
#         from pdf2image import convert_from_path
#         from transformers import pipeline
#         from pathlib import Path
#         dir_ = Path(__file__).parent

#         p = pipeline(
#             "document-question-answering",model="impira/layoutlm-document-qa",
#         )

#         def qa(question: str, doc: str) -> str:
#             img = convert_from_path(doc)[0]
#             output = p(img, question)
#             return sorted(output, key=lambda x: x["score"], reverse=True)[0]['answer']

# demo = gr.Interface(
#     qa,
#     [gr.Textbox(label="Question"), PDF(label="Document")],
#     gr.Textbox(),
#     examples=[["What is the total gross worth?", str(dir_ / "invoice_2.pdf")],
#             ["Whos is being invoiced?", str(dir_ / "sample_invoice.pdf")]]
# )



if __name__ == "__main__":
    summarizer = GeneralTextSummarizer(
        model_name=SummarizerModelSize.BASE,
        use_cuda=torch.cuda.is_available(),
        batch_stride=16,
        token_batch_length=2048,
        compile_model=False,
        min_length=8,
        max_length=256,
    )
    # example1 = "the big variety of data coming from diverse sources is one of the key properties of the big data phenomenon. It is, therefore, beneficial to understand how data is generated in various environments and scenarios, before looking at what should be done with this data and how to design the best possible architecture to accomplish this The evolution of IT architectures, described in Chapter 2, means that the data is no longer processed by a few big monolith systems, but rather by a group of services In parallel to the processing layer, the underlying data storage has also changed and became more distributed This, in turn, required a significant paradigm shift as the traditional approach to transactions (ACID) could no longer be supported. On top of this, cloud computing is becoming a major approach with the benefits of reducing costs and providing on-demand scalability but at the same time introducing concerns about privacy, data ownership, etc In the meantime the Internet continues its exponential growth: Every day both structured and unstructured data is published and available for processing: To achieve competitive advantage companies have to relate their corporate resources to external services, e.g. financial markets, weather forecasts, social media, etc While several of the sites provide some sort of API to access the data in a more orderly fashion; countless sources require advanced web mining and Natural Language Processing (NLP) processing techniques: Advances in science push researchers to construct new instruments for observing the universe O conducting experiments to understand even better the laws of physics and other domains. Every year humans have at their disposal new telescopes, space probes, particle accelerators, etc These instruments generate huge streams of data, which need to be stored and analyzed. The constant drive for efficiency in the industry motivates the introduction of new automation techniques and process optimization: This could not be done without analyzing the precise data that describe these processes. As more and more human tasks are automated, machines provide rich data sets, which can be analyzed in real-time to drive efficiency to new levels. Finally, it is now evident that the growth of the Internet of Things is becoming a major source of data. More and more of the devices are equipped with significant computational power and can generate a continuous data stream from their sensors. In the subsequent sections of this chapter, we will look at the domains described above to see what they generate in terms of data sets. We will compare the volumes but will also look at what is characteristic and important from their respective points of view. 3.1 The Internet is undoubtedly the largest database ever created by humans. While several well described; cleaned, and structured data sets have been made available through this medium, most of the resources are of an ambiguous, unstructured, incomplete or even erroneous nature. Still, several examples in the areas such as opinion mining, social media analysis, e-governance, etc, clearly show the potential lying in these resources. Those who can successfully mine and interpret the Internet data can gain unique insight and competitive advantage in their business An important area of data analytics on the edge of corporate IT and the Internet is Web Analytics."
    # result = summarizer.summarize_text(example1)
    # print(result)
    result = summarizer.summarize_filetype("/home/pop/Downloads/japan_outlook_gor2401b-1.pdf")
    print(result)


    """  
    ## USAGE 

    summarizer = GeneralTextSummarizer(
        model_name=SummarizerModelSize.BASE,
        use_cuda=True,
        batch_stride=16,
        token_batch_length=2048,
        compile_model=False,
        min_length=8,
        max_length=256,
    )

    example1 = "the big variety of data coming from diverse sources is one of the key properties of the big data phenomenon. It is, therefore, beneficial to understand how data is generated in various environments and scenarios, before looking at what should be done with this data and how to design the best possible architecture to accomplish this The evolution of IT architectures, described in Chapter 2, means that the data is no longer processed by a few big monolith systems, but rather by a group of services In parallel to the processing layer, the underlying data storage has also changed and became more distributed This, in turn, required a significant paradigm shift as the traditional approach to transactions (ACID) could no longer be supported. On top of this, cloud computing is becoming a major approach with the benefits of reducing costs and providing on-demand scalability but at the same time introducing concerns about privacy, data ownership, etc In the meantime the Internet continues its exponential growth: Every day both structured and unstructured data is published and available for processing: To achieve competitive advantage companies have to relate their corporate resources to external services, e.g. financial markets, weather forecasts, social media, etc While several of the sites provide some sort of API to access the data in a more orderly fashion; countless sources require advanced web mining and Natural Language Processing (NLP) processing techniques: Advances in science push researchers to construct new instruments for observing the universe O conducting experiments to understand even better the laws of physics and other domains. Every year humans have at their disposal new telescopes, space probes, particle accelerators, etc These instruments generate huge streams of data, which need to be stored and analyzed. The constant drive for efficiency in the industry motivates the introduction of new automation techniques and process optimization: This could not be done without analyzing the precise data that describe these processes. As more and more human tasks are automated, machines provide rich data sets, which can be analyzed in real-time to drive efficiency to new levels. Finally, it is now evident that the growth of the Internet of Things is becoming a major source of data. More and more of the devices are equipped with significant computational power and can generate a continuous data stream from their sensors. In the subsequent sections of this chapter, we will look at the domains described above to see what they generate in terms of data sets. We will compare the volumes but will also look at what is characteristic and important from their respective points of view. 3.1 The Internet is undoubtedly the largest database ever created by humans. While several well described; cleaned, and structured data sets have been made available through this medium, most of the resources are of an ambiguous, unstructured, incomplete or even erroneous nature. Still, several examples in the areas such as opinion mining, social media analysis, e-governance, etc, clearly show the potential lying in these resources. Those who can successfully mine and interpret the Internet data can gain unique insight and competitive advantage in their business An important area of data analytics on the edge of corporate IT and the Internet is Web Analytics."
    summarizer.summarize_text(example1)
    """
