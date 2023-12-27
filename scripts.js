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

            // Set the text content of the div
            div.textContent = `Name: ${item.name}, Lowest Bin: ${item.lowest_bin}, Second Lowest Bin: ${item.second_lowest_bin}, Profit: ${item.profit}, Profit Percentage: ${item.profit_percentage}, Items in AH: ${item.items_in_ah}, Item Category: ${item.item_category}`;

            // Append the div to the #auctions section
            auctionsSection.appendChild(div);
        });
    })
    .catch(error => console.error('Error:', error));