// Fetch the data from the local server
fetch('http://localhost:8000/auction_results.json')
    .then(response => response.json()) // Convert the response to a JavaScript object
    .then(data => {
        // Get the #auctions section
        const auctionsSection = document.querySelector('#auctions');

        // Loop over the array of objects
        data.forEach(item => {
            // Create a new div element
            const div = document.createElement('div');

            // Add a class to the div
            div.classList.add('item-box');

            // Set the text content of the div
            div.innerHTML = `
                <h2>${item.name}</h2>
                <p>Lowest Bin: ${item.lowest_bin}</p>
                <p>Second Lowest Bin: ${item.second_lowest_bin}</p>
                <p>Profit: ${item.profit}</p>
                <p>Profit Percentage: ${item.profit_percentage}</p>
                <p>Items in AH: ${item.items_in_ah}</p>
                <p>Item Category: ${item.item_category}</p>
            `;

            // Append the div to the #auctions section
            auctionsSection.appendChild(div);
        });
    })
    .catch(error => console.error('Error:', error));