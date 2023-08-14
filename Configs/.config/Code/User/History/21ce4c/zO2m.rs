use yew::prelude::*;

struct team {
    name: String,
    points: i32,
}
struct score {
    team1: team,
    team2: team,
    team3: team,
    team4: team,
}

#[function_component(App)]
fn app() -> Html {
    let team1 = team {
        name: "Garlic Gang".to_string(),
        points: 0,
    };
    let team2 = team {
        name: "Cat-chup".to_string(),
        points: 0,
    };
    let team3 = team {
        name: "The 3 Musketeers".to_string(),
        points: 0,
    };
    let team4 = team {
        name: "The 4 Musketeers".to_string(),
        points: 0,
    };

    html! {
    html! {
        <>
        <div class="container">
        <div class="item">
        <div class="teamName" id="Team1">{team1.name}</div>
        <div class="punkte" id="Punkte1">{"0"}</div>
    </div>
    <div class="item">
        <div class="teamName" id="Team2">{"Team 2"}</div>
        <div class="punkte" id="Punkte2">{"0"}</div>
        </div>
        <div class="item">
        <div class="teamName" id="Team3">{"Team 3"}</div>
        <div class="punkte" id="Punkte3">{"0"}</div>
        </div>
        <div class="item">
        <div class="teamName" id="Team4">{"Team 4"}</div>
        <div class="punkte" id="Punkte4">{"0"}</div>
        </div>
        </div>
        </>
    }    }
}

fn main() {
    yew::Renderer::<App>::new().render();
}
