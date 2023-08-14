use yew::prelude::*;

#[function_component(App)]
fn app() -> Html {
    html! {
    html! {
        <>
        <div class="container">
        <div class="item">
        <div class="teamName" id="Team1">{"Team 1"}</div>
        <div class="punkte" id="Punkte1">{"0"}</div>
    </div>
        </div>
        </>
    }    }
}

fn main() {
    yew::Renderer::<App>::new().render();
}
