use std::collections::HashMap;

use axum::{routing::get, Router};
struct Team {
    name: String,
    score: i32,
}
struct scoreboard {
    team1: Team,
    team2: Team,
    team3: Team,
    team4: Team,
}

#[tokio::main]
async fn main() {
    let mut team1 = Team {
        name: "Team 1".to_string(),
        score: 0,
    };
    let mut team2 = Team {
        name: "Team 2".to_string(),
        score: 0,
    };
    let mut team3 = Team {
        name: "Team 3".to_string(),
        score: 0,
    };
    let mut team4 = Team {
        name: "Team 4".to_string(),
        score: 0,
    };
    let mut scoreboard = scoreboard {
        team1: team1,
        team2: team2,
        team3: team3,
        team4: team4,
    };

    // serve index.html in assets directory using axum
    let app = route("/", get(|| async { Html(include_str!("index.html").to_string()) }))
    .route("/update", post(update_team))
    .layer(axum::AddExtensionLayer::new(shared_team));
}
