// when an element of the class item is clicked, the function is called.
// get the id of the element and pass it to the function
// the function will then increase the points of the team by 1

document.addEventListener('click', function (event) {
    if (event.target.parentElement.classList.contains('item')) {

        // switch:  if the id of the element is team1, then increase the points of team1
        // if the id of the element is team2, then increase the points of team2
        switch (event.target.parentElement.id) {
            case 'team1':
                increasePoints('team1');
                break;
            case 'team2':
                increasePoints('team2');
                break;
            case 'team3':
                increasePoints('team3');
                break;
            case 'team4':
                increasePoints('team4');
                break;
        }
    }
});