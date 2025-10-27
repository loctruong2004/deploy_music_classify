# ui_navbar.py
import gradio as gr

def add_navbar(active: str = "app"):
    variants = {"login": "secondary", "app": "secondary", "player": "secondary"}
    if active in variants:
        variants[active] = "primary"

    with gr.Row(elem_id="top-navbar"):
        btn_app    = gr.Button("App",    variant=variants["app"])
        btn_player = gr.Button("Player", variant=variants["player"])
        btn_logout = gr.Button("Logout", variant="destructive")

    btn_app.click(None,   [], [], js="()=>{ try{window.top.location.href='/app'}catch(e){window.location.href='/app'} }")
    btn_player.click(None,[], [], js="()=>{ try{window.top.location.href='/player'}catch(e){window.location.href='/player'} }")
    btn_logout.click(
        None, [], [], 
        js="""
        () => {
          fetch('/auth/logout', {method:'POST'}).then(()=> {
            try{ window.top.location.href='/login'; } catch(e){ window.location.href='/login'; }
          });
        }
        """
    )
