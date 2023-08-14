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

    let index_file = ServeDir::new("assets/index.html");
    // build our application with a single route
    let app = Router::new().route("/", get(|| async { "Hello, World!" }));

    // run it with hyper on localhost:3000
    axum::Server::bind(&"0.0.0.0:3001".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}
