import gradio as gr
import torch
from transformers import (
    pipeline,
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)

M0 = "consciousAI/question-generation-auto-t5-v1-base-s"
M1 = "consciousAI/question-generation-auto-t5-v1-base-s-q"
M2 = "consciousAI/question-generation-auto-t5-v1-base-s-q-c" 

M4 = "consciousAI/question-generation-auto-hints-t5-v1-base-s-q"
M5 = "consciousAI/question-generation-auto-hints-t5-v1-base-s-q-c" 

device = ['cuda' if torch.cuda.is_available() else 'cpu'][0]

_m0 = AutoModelForSeq2SeqLM.from_pretrained(M0).to(device)
_tk0 = AutoTokenizer.from_pretrained(M0, cache_dir="./cache")

_m1 = AutoModelForSeq2SeqLM.from_pretrained(M1).to(device)
_tk1 = AutoTokenizer.from_pretrained(M1, cache_dir="./cache")

_m2 = AutoModelForSeq2SeqLM.from_pretrained(M2).to(device)
_tk2 = AutoTokenizer.from_pretrained(M2, cache_dir="./cache")

_m4 = AutoModelForSeq2SeqLM.from_pretrained(M4).to(device)
_tk4 = AutoTokenizer.from_pretrained(M4, cache_dir="./cache")

_m5 = AutoModelForSeq2SeqLM.from_pretrained(M5).to(device)
_tk5 = AutoTokenizer.from_pretrained(M5, cache_dir="./cache")

def _formatQs(questions):
    _finalQs = ""
    
    if questions is not None:
        _qList = questions[0].strip().split("?")
        
        qIdx = 1
        if len(_qList) > 1:
            for idx, _q in enumerate(_qList):
                _q = _q.strip()
                if _q is not None and len(_q) !=0:
                    _finalQs += str(qIdx) + ". " + _q + "? \n"
                    qIdx+=1
        else:
            if len(_qList[0])>1:
                _finalQs = "1. " + str(_qList[0]) + "?"
            else:
                _finalQs = None
    return _finalQs
    
def _generate(mode, context, hint=None, minLength=50, maxLength=500, lengthPenalty=2.0, earlyStopping=True, numReturnSequences=1, numBeams=2, noRepeatNGramSize=0, doSample=False, topK=0, penaltyAlpha=0, topP=0, temperature=0, model="All"):
          
    predictionM0 = None
    predictionM1 = None
    predictionM2 = None
    predictionM4 = None
    predictionM5 = None
    
    if mode == 'Auto':
        _inputText = "question_context: " + context
        
        if model == "All": 
            _encoding = _tk0.encode(_inputText, return_tensors='pt', truncation=True, padding='max_length').to(device) # max_length=1024 
            _outputEncoded = _m0.generate(_encoding, 
                                       min_length=minLength, 
                                       max_length=maxLength,
                                       length_penalty=lengthPenalty,
                                       early_stopping=earlyStopping,
                                       num_return_sequences=numReturnSequences,
                                       num_beams=numBeams,
                                       no_repeat_ngram_size=noRepeatNGramSize,
                                       do_sample=doSample,
                                       top_k=topK,
                                       penalty_alpha=penaltyAlpha,
                                       top_p=topP,
                                       temperature=temperature
                                   )
            predictionM0 = [_tk0.decode(id, clean_up_tokenization_spaces=False, skip_special_tokens=True) for id in _outputEncoded]

            _encoding = _tk1.encode(_inputText, return_tensors='pt', truncation=True, padding='max_length').to(device) # max_length=1024 
            _outputEncoded = _m1.generate(_encoding, 
                                       min_length=minLength, 
                                       max_length=maxLength,
                                       length_penalty=lengthPenalty,
                                       early_stopping=earlyStopping,
                                       num_return_sequences=numReturnSequences,
                                       num_beams=numBeams,
                                       no_repeat_ngram_size=noRepeatNGramSize,
                                       do_sample=doSample,
                                       top_k=topK,
                                       penalty_alpha=penaltyAlpha,
                                       top_p=topP,
                                       temperature=temperature
                                   )
            predictionM1 = [_tk1.decode(id, clean_up_tokenization_spaces=False, skip_special_tokens=True) for id in _outputEncoded]

            _encoding = _tk2.encode(_inputText, return_tensors='pt', truncation=True, padding='max_length').to(device) # max_length=1024 .to(device)
            _outputEncoded = _m2.generate(_encoding, 
                                       min_length=minLength, 
                                       max_length=maxLength,
                                       length_penalty=lengthPenalty,
                                       early_stopping=earlyStopping,
                                       num_return_sequences=numReturnSequences,
                                       num_beams=numBeams,
                                       no_repeat_ngram_size=noRepeatNGramSize,
                                       do_sample=doSample,
                                       top_k=topK,
                                       penalty_alpha=penaltyAlpha,
                                       top_p=topP,
                                       temperature=temperature
                                   )
            predictionM2 = [_tk2.decode(id, clean_up_tokenization_spaces=False, skip_special_tokens=True) for id in _outputEncoded]

            _encoding = _tk4.encode(_inputText, return_tensors='pt', truncation=True, padding='max_length').to(device) # max_length=1024 .to(device)
            _outputEncoded = _m4.generate(_encoding, 
                                       min_length=minLength, 
                                       max_length=maxLength,
                                       length_penalty=lengthPenalty,
                                       early_stopping=earlyStopping,
                                       num_return_sequences=numReturnSequences,
                                       num_beams=numBeams,
                                       no_repeat_ngram_size=noRepeatNGramSize,
                                       do_sample=doSample,
                                       top_k=topK,
                                       penalty_alpha=penaltyAlpha,
                                       top_p=topP,
                                       temperature=temperature
                                   )
            predictionM4 = [_tk4.decode(id, clean_up_tokenization_spaces=False, skip_special_tokens=True) for id in _outputEncoded]

            _encoding = _tk5.encode(_inputText, return_tensors='pt', truncation=True, padding='max_length').to(device) # max_length=1024 .to(device)
            _outputEncoded = _m5.generate(_encoding, 
                                       min_length=minLength, 
                                       max_length=maxLength,
                                       length_penalty=lengthPenalty,
                                       early_stopping=earlyStopping,
                                       num_return_sequences=numReturnSequences,
                                       num_beams=numBeams,
                                       no_repeat_ngram_size=noRepeatNGramSize,
                                       do_sample=doSample,
                                       top_k=topK,
                                       penalty_alpha=penaltyAlpha,
                                       top_p=topP,
                                       temperature=temperature
                                   )
            predictionM5 = [_tk5.decode(id, clean_up_tokenization_spaces=False, skip_special_tokens=True) for id in _outputEncoded]
        elif model == "question-generation-auto-hints-t5-v1-base-s-q-c":
            _encoding = _tk5.encode(_inputText, return_tensors='pt', truncation=True, padding='max_length').to(device) # max_length=1024 .to(device)
            _outputEncoded = _m5.generate(_encoding, 
                                       min_length=minLength, 
                                       max_length=maxLength,
                                       length_penalty=lengthPenalty,
                                       early_stopping=earlyStopping,
                                       num_return_sequences=numReturnSequences,
                                       num_beams=numBeams,
                                       no_repeat_ngram_size=noRepeatNGramSize,
                                       do_sample=doSample,
                                       top_k=topK,
                                       penalty_alpha=penaltyAlpha,
                                       top_p=topP,
                                       temperature=temperature
                                   )
            predictionM5 = [_tk5.decode(id, clean_up_tokenization_spaces=False, skip_special_tokens=True) for id in _outputEncoded]
        elif model == "question-generation-auto-hints-t5-v1-base-s-q":
            _encoding = _tk4.encode(_inputText, return_tensors='pt', truncation=True, padding='max_length').to(device) # max_length=1024 .to(device)
            _outputEncoded = _m4.generate(_encoding, 
                                       min_length=minLength, 
                                       max_length=maxLength,
                                       length_penalty=lengthPenalty,
                                       early_stopping=earlyStopping,
                                       num_return_sequences=numReturnSequences,
                                       num_beams=numBeams,
                                       no_repeat_ngram_size=noRepeatNGramSize,
                                       do_sample=doSample,
                                       top_k=topK,
                                       penalty_alpha=penaltyAlpha,
                                       top_p=topP,
                                       temperature=temperature
                                   )
            predictionM4 = [_tk4.decode(id, clean_up_tokenization_spaces=False, skip_special_tokens=True) for id in _outputEncoded]
        elif model == "question-generation-auto-t5-v1-base-s-q-c":
            _encoding = _tk2.encode(_inputText, return_tensors='pt', truncation=True, padding='max_length').to(device) # max_length=1024 .to(device)
            _outputEncoded = _m2.generate(_encoding, 
                                       min_length=minLength, 
                                       max_length=maxLength,
                                       length_penalty=lengthPenalty,
                                       early_stopping=earlyStopping,
                                       num_return_sequences=numReturnSequences,
                                       num_beams=numBeams,
                                       no_repeat_ngram_size=noRepeatNGramSize,
                                       do_sample=doSample,
                                       top_k=topK,
                                       penalty_alpha=penaltyAlpha,
                                       top_p=topP,
                                       temperature=temperature
                                   )
            predictionM2 = [_tk2.decode(id, clean_up_tokenization_spaces=False, skip_special_tokens=True) for id in _outputEncoded]
        elif model == "question-generation-auto-t5-v1-base-s-q":
            _encoding = _tk1.encode(_inputText, return_tensors='pt', truncation=True, padding='max_length').to(device) # max_length=1024 
            _outputEncoded = _m1.generate(_encoding, 
                                       min_length=minLength, 
                                       max_length=maxLength,
                                       length_penalty=lengthPenalty,
                                       early_stopping=earlyStopping,
                                       num_return_sequences=numReturnSequences,
                                       num_beams=numBeams,
                                       no_repeat_ngram_size=noRepeatNGramSize,
                                       do_sample=doSample,
                                       top_k=topK,
                                       penalty_alpha=penaltyAlpha,
                                       top_p=topP,
                                       temperature=temperature
                                   )
            predictionM1 = [_tk1.decode(id, clean_up_tokenization_spaces=False, skip_special_tokens=True) for id in _outputEncoded]
        elif model == "question-generation-auto-t5-v1-base-s":
            _encoding = _tk0.encode(_inputText, return_tensors='pt', truncation=True, padding='max_length').to(device) # max_length=1024 
            _outputEncoded = _m0.generate(_encoding, 
                                       min_length=minLength, 
                                       max_length=maxLength,
                                       length_penalty=lengthPenalty,
                                       early_stopping=earlyStopping,
                                       num_return_sequences=numReturnSequences,
                                       num_beams=numBeams,
                                       no_repeat_ngram_size=noRepeatNGramSize,
                                       do_sample=doSample,
                                       top_k=topK,
                                       penalty_alpha=penaltyAlpha,
                                       top_p=topP,
                                       temperature=temperature
                                   )
            predictionM0 = [_tk0.decode(id, clean_up_tokenization_spaces=False, skip_special_tokens=True) for id in _outputEncoded]
    elif mode == 'Hints':
        _inputText = "question_hint: " + hint + "</s>question_context: " + context

        _encoding = _tk4.encode(_inputText, return_tensors='pt', truncation=True, padding='max_length').to(device) # max_length=1024 .to(device)
        _outputEncoded = _m4.generate(_encoding, 
                                   min_length=minLength, 
                                   max_length=maxLength,
                                   length_penalty=lengthPenalty,
                                   early_stopping=earlyStopping,
                                   num_return_sequences=numReturnSequences,
                                   num_beams=numBeams,
                                   no_repeat_ngram_size=noRepeatNGramSize,
                                   do_sample=doSample,
                                   top_k=topK,
                                   penalty_alpha=penaltyAlpha,
                                   top_p=topP,
                                   temperature=temperature
                               )
        predictionM4 = [_tk4.decode(id, clean_up_tokenization_spaces=False, skip_special_tokens=True) for id in _outputEncoded]
        
        _encoding = _tk5.encode(_inputText, return_tensors='pt', truncation=True, padding='max_length').to(device) # max_length=1024 .to(device)
        _outputEncoded = _m5.generate(_encoding, 
                                   min_length=minLength, 
                                   max_length=maxLength,
                                   length_penalty=lengthPenalty,
                                   early_stopping=earlyStopping,
                                   num_return_sequences=numReturnSequences,
                                   num_beams=numBeams,
                                   no_repeat_ngram_size=noRepeatNGramSize,
                                   do_sample=doSample,
                                   top_k=topK,
                                   penalty_alpha=penaltyAlpha,
                                   top_p=topP,
                                   temperature=temperature
                               )
        predictionM5 = [_tk5.decode(id, clean_up_tokenization_spaces=False, skip_special_tokens=True) for id in _outputEncoded]
  
    predictionM0 = _formatQs(predictionM0)
    predictionM1 = _formatQs(predictionM1)
    predictionM2 = _formatQs(predictionM2)
    predictionM4 = _formatQs(predictionM4)
    predictionM5 = _formatQs(predictionM5)

    return predictionM5, predictionM4, predictionM2, predictionM1, predictionM0
    
with gr.Blocks() as demo:
    gr.Markdown(value="# Question Generation Demo \n [question-generation-auto-t5-v1-base-s](https://huggingface.co/anshoomehra/question-generation-auto-t5-v1-base-s) ✫ [question-generation-auto-t5-v1-base-s-q](https://huggingface.co/anshoomehra/question-generation-auto-t5-v1-base-s-q) ✫ [question-generation-auto-t5-v1-base-s-q-c](https://huggingface.co/anshoomehra/question-generation-auto-t5-v1-base-s-q-c) ✫ [question-generation-auto-hints-t5-v1-base-s-q](https://huggingface.co/anshoomehra/question-generation-auto-hints-t5-v1-base-s-q) ✫ [question-generation-auto-hints-t5-v1-base-s-q-c](https://huggingface.co/anshoomehra/question-generation-auto-hints-t5-v1-base-s-q-c)\n\n Please be patient, 5 models may take up to 80 sec to run on CPU")

    with gr.Accordion(variant='compact', label='Search Methods: Deteriminstic / Stochastic / Contrastive', open=True):
        with gr.Row():
            mode = gr.Radio(["Auto", "Hints"], value="Auto", label="Mode")
        with gr.Row():   
            minLength = gr.Slider(10, 512, 50, step=1, label="Min Length")
            maxLength = gr.Slider(20, 512, 164, step=1, label="Max Length")
            lengthPenalty = gr.Slider(-5, 5, 1, label="Length Penalty")
            earlyStopping = gr.Checkbox(True, label="Early Stopping [EOS]")
            numReturnSequences = gr.Slider(1, 3, 1, step=1, label="Num return Sequences")
        with gr.Row():   
            numBeams = gr.Slider(1, 10, 4, step=1, label="Beams")
            noRepeatNGramSize = gr.Slider(0, 5, 3, step=1, label="No Repeat N-Gram Size")
        with gr.Row():
            doSample = gr.Checkbox(label="Do Random Sample")
            topK = gr.Slider(0, 50, 0, step=1, label="Top K")
            penaltyAlpha = gr.Slider(0.0, 1, 0, label="Penalty Alpha")
            topP = gr.Slider(0, 1, 0, label="Top P/Nucleus Sampling")
            temperature = gr.Slider(0.01, 1, 1, label="Temperature") 
        with gr.Row():
            model = gr.Dropdown(["question-generation-auto-hints-t5-v1-base-s-q-c", "question-generation-auto-hints-t5-v1-base-s-q", "question-generation-auto-t5-v1-base-s-q-c", "question-generation-auto-t5-v1-base-s-q", "question-generation-auto-t5-v1-base-s", "All"], label="Model", value="question-generation-auto-hints-t5-v1-base-s-q-c")    
         
    
    with gr.Accordion(variant='compact', label='Input Values'):
        with gr.Row(variant='compact'):
                contextDefault = "Google LLC is an American multinational technology company focusing on search engine technology, online advertising, cloud computing, computer software, quantum computing, e-commerce, artificial intelligence, and consumer electronics. It has been referred to as 'the most powerful company in the world' and one of the world's most valuable brands due to its market dominance, data collection, and technological advantages in the area of artificial intelligence. Its parent company Alphabet is considered one of the Big Five American information technology companies, alongside Amazon, Apple, Meta, and Microsoft."
                hintDefault  = ""
                context = gr.Textbox(contextDefault, label="Context", placeholder="Dummy Context", lines=5)
                hint = gr.Textbox(hintDefault, label="Hint", placeholder="Enter hint here. Ensure the mode is set to 'Hints' prior using hints.", lines=2)
 
    with gr.Accordion(variant='compact', label='Multi-Task Model(s) Sensitive To Hints'):
        with gr.Row(variant='compact'):
            _predictionM5 = gr.Textbox(label="Predicted Questions - question-generation-auto-hints-t5-v1-base-s-q-c [Hints Sensitive]")
            _predictionM4 = gr.Textbox(label="Predicted Questions - question-generation-auto-hints-t5-v1-base-s-q [Hints Sensitive]")
            
    with gr.Accordion(variant='compact', label='Uni-Task Model(s) Non-Sensitive To Hints'):
        with gr.Row(variant='compact'):
            _predictionM2 = gr.Textbox(label="Predicted Questions - question-generation-auto-t5-v1-base-s-q-c [No Hints]")
            _predictionM1 = gr.Textbox(label="Predicted Questions - question-generation-auto-t5-v1-base-s-q [No Hints]")
            _predictionM0 = gr.Textbox(label="Predicted Questions - question-generation-auto-t5-v1-base-s [No Hints]")

    with gr.Row():       
        gen_btn = gr.Button("Generate Questions")
        gen_btn.click(fn=_generate,
                      inputs=[mode, context, hint, minLength, maxLength, lengthPenalty, earlyStopping, numReturnSequences, numBeams, noRepeatNGramSize, doSample, topK, penaltyAlpha, topP, temperature, model],
                      outputs=[_predictionM5, _predictionM4, _predictionM2, _predictionM1, _predictionM0]
                      )

demo.launch(show_error=True)
