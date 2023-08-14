// when an element of the class item is clicked, the function is called.
// get the id of the element and pass it to the function
// the function will then increase the points of the team by 1

document.addEventListener('click', function (event) {
    if (event.target.parentElement.classList.contains('item')) {

        // switch:  if the id of the element is team1, then increase the points of team1
        // if the id of the element is team2, then increase the points of team2
        switch (event.target.parentElement.id) {
            case 'item1':
                increasePoints('punkte1');
                break;
            case 'item2':
                increasePoints('punkte2');
                break;
            case 'item3':
                increasePoints('punkte3');
                break;
            case 'item4':
                increasePoints('punkte4');
                break;
        }
    }
});

function increasePoints(team) {
    // get the points of the team
    let points = document.getElementById(team).innerHTML;
    initConfetti();

    // increase the points by 1
    points++;

    // update the points of the team
    document.getElementById(team).innerHTML = points;
}
    //-----------Var Inits--------------
    canvas = document.getElementById("canvas");
    ctx = canvas.getContext("2d");
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    cx = ctx.canvas.width / 2;
    cy = ctx.canvas.height / 2;
     
    let confetti = [];
    const confettiCount = 500;
    const gravity = 1;
    const terminalVelocity = 6;
    const drag = 0.075;
    const colors = [
    { front: 'red', back: 'darkred' },
    { front: 'green', back: 'darkgreen' },
    { front: 'blue', back: 'darkblue' },
    { front: 'yellow', back: 'darkyellow' },
    { front: 'orange', back: 'darkorange' },
    { front: 'pink', back: 'darkpink' },
    { front: 'purple', back: 'darkpurple' },
    { front: 'turquoise', back: 'darkturquoise' }];
     
     
    //-----------Functions--------------
    resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      cx = ctx.canvas.width / 2;
      cy = ctx.canvas.height / 2;
    };
     
    randomRange = (min, max) => Math.random() * (max - min) + min;
     
    initConfetti = () => {
      for (let i = 0; i < confettiCount; i++) {
        confetti.push({
          color: colors[Math.floor(randomRange(0, colors.length))],
          dimensions: {
            x: randomRange(10, 20),
            y: randomRange(10, 30) },
     
          position: {
            x: randomRange(0, canvas.width),
            y: canvas.height - 1 },
     
          rotation: randomRange(0, 2 * Math.PI),
          scale: {
            x: 1,
            y: 1 },
     
          velocity: {
            x: randomRange(-25, 25),
            y: randomRange(0, -50) } });
     
     
      }
    };
     
    //---------Render-----------
    render = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
     
      confetti.forEach((confetto, index) => {
        let width = confetto.dimensions.x * confetto.scale.x;
        let height = confetto.dimensions.y * confetto.scale.y;
     
        // Move canvas to position and rotate
        ctx.translate(confetto.position.x, confetto.position.y);
        ctx.rotate(confetto.rotation);
     
        // Apply forces to velocity
        confetto.velocity.x -= confetto.velocity.x * drag;
        confetto.velocity.y = Math.min(confetto.velocity.y + gravity, terminalVelocity);
        confetto.velocity.x += Math.random() > 0.5 ? Math.random() : -Math.random();
     
        // Set position
        confetto.position.x += confetto.velocity.x;
        confetto.position.y += confetto.velocity.y;
     
        // Delete confetti when out of frame
        if (confetto.position.y >= canvas.height) confetti.splice(index, 1);
     
        // Loop confetto x position
        if (confetto.position.x > canvas.width) confetto.position.x = 0;
        if (confetto.position.x < 0) confetto.position.x = canvas.width;
     
        // Spin confetto by scaling y
        confetto.scale.y = Math.cos(confetto.position.y * 0.1);
        ctx.fillStyle = confetto.scale.y > 0 ? confetto.color.front : confetto.color.back;
     
        // Draw confetti
        ctx.fillRect(-width / 2, -height / 2, width, height);
     
        // Reset transform matrix
        ctx.setTransform(1, 0, 0, 1, 0, 0);
      });
     
      window.requestAnimationFrame(render);
    };
     
    //---------Execution--------
    render();
     
    //----------Resize----------
    window.addEventListener('resize', function () {
      resizeCanvas();
    });
     