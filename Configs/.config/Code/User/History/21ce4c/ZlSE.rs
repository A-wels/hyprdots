use std::collections::HashMap;

use axum::{
    routing::get,
    Router,
};
struct Team{
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
    let mut scoreboard = scoreboard {
        scoreTeam1: 0,
        scoreTeam2: 0,
        scoreTeam3: 0,
        scoreTeam4: 0,
    };

    // build our application with a single route
    let app = Router::new().route("/", get(|| async { "Hello, World!" }));

    // run it with hyper on localhost:3000
    axum::Server::bind(&"0.0.0.0:3001".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}