import dash
from dash import dcc, html, Input, Output, State
import openai

# Set up the OpenAI API
openai.api_key = "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"

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Chat with AI"),
    dcc.Textarea(
        id="chat-input",
        placeholder="Type your message here...",
        style={"width": "100%", "height": "100px", "font-size": "16px"}
    ),
    html.Button("Send", id="send-button", n_clicks=0),
    html.Div(id="chat-output", style={"white-space": "pre-wrap", "font-size": "16px"})
])

@app.callback(
    Output("chat-output", "children"),
    [Input("send-button", "n_clicks")],
    [State("chat-input", "value")])
def send_message(n_clicks, message):
    if n_clicks > 0 and message:
        
        response = get_gpt_response(message)
        print(response)
        return f"You: {message}\nAI: {response}"
    return ""

def get_gpt_response(prompt):

    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}])
    
    return response.choices[0].message.content

if __name__ == "__main__":
    app.run_server(debug=True)