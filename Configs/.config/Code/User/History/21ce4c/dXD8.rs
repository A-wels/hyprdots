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
        <div class="punkte" id="Punkte1">{team1.points}</div>
    </div>
    <div class="item">
        <div class="teamName" id="Team2">{team2.name}</div>
        <div class="punkte" id="Punkte2">{team2.points}</div>
        </div>
        <div class="item">
        <div class="teamName" id="Team3">{team3.name}</div>
        <div class="punkte" id="Punkte3">{team3.points}</div>
        </div>
        <div class="item">
        <div class="teamName" id="Team4">{team4.name}</div>
        <div class="punkte" id="Punkte4">{team4.points}</div>
        </div>
        </div>
        </>
    }    }
}

fn main() {
    yew::Renderer::<App>::new().render();
}