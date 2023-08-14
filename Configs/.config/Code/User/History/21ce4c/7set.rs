use yew::prelude::*;

#[function_component(App)]
fn app() -> Html {
    html! {
        html! {
            <>
            <div class="container">
            </div>
            </>
        }    }
}

fn main() {
    yew::Renderer::<App>::new().render();
}
