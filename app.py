import gradio as gr
from service import Service

def doctor_bot(message, history):
    service = Service()
    result = service.answer(message, history)
    return result['output']

css = '''
.gradio-container { max-width:850px !important; margin:20px auto !important;}
.message { padding: 10px !important; font-size: 14px !important;}
'''

demo = gr.ChatInterface(
    css = css,
    fn = doctor_bot, 
    title = 'Doctor Bot',
    chatbot = gr.Chatbot(height=400, bubble_full_width=False),
    theme = gr.themes.Default(spacing_size='sm', radius_size='sm'),
    textbox=gr.Textbox(placeholder="Enter your question here", container=False, scale=7),
    examples = ['What is your name?', 'What causes a cold?' ,'How to treat rhinitis?','What are symptoms of rhnitis?','Are rhinitis and colds complications?', 'What medicine can cure a cold quickly? Can I take amoxicillin?'],
    submit_btn = gr.Button('Submit', variant='primary'),
    clear_btn = gr.Button('Clear'),
    retry_btn = None,
    undo_btn = None,
)

if __name__ == '__main__':
    demo.launch()