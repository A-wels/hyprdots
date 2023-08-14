use yew::prelude::*;

#[function_component(App)]
fn app() -> Html {
    html! {
        html! {
            <>
            <div class="container">
            <div class="item">
                <div class="teamName" id="Team1">{Team 1}</div>
                <div class="punkte" id="Punkte1">{0}</div>
            </div>
           <div class="item">
                <div class="teamName" id="Team2">{Team 2}</div>
                <div class="punkte" id="Punkte2">{0}</div>
            </div>
            <div class="item">
                <div class="teamName" id="Team3">{Team 3}</div>
                <div class="punkte" id="Punkte3">{0}</div>
            </div>
            <div class="item">
                <div class="teamName" id="Team4">{Team 4}</div>
                <div class="punkte" id="Punkte4">{0}</div>
            </>
        }    }
}

fn main() {
    yew::Renderer::<App>::new().render();
}
