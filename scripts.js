document.addEventListener("DOMContentLoaded", function () {
    // Function to fetch auction data from the server
    function fetchAuctionData() {
        // Make an AJAX request to your server endpoint
        // Modify the URL to match the endpoint where your Python server serves auction data
        fetch('https://your-server-endpoint.com/auction_data')
            .then(response => response.json())
            .then(data => {
                // Process fetched data and display it on the webpage
                displayAuctionData(data);
            })
            .catch(error => {
                console.error('Error fetching auction data:', error);
            });
    }

    // Function to display auction data on the webpage
    function displayAuctionData(data) {
        const auctionsSection = document.getElementById('auctions');

        // Clear previous content
        auctionsSection.innerHTML = '';

        // Create HTML elements to display auction data
        data.forEach(auction => {
            const auctionElement = document.createElement('div');
            auctionElement.classList.add('auction-item');

            // Customize the content based on your data structure
            auctionElement.innerHTML = `
                <h2>${auction.name}</h2>
                <p>Lowest Bin: ${auction.lowest_bin}</p>
                <p>Second Lowest Bin: ${auction.second_lowest_bin}</p>
                <p>Profit: ${auction.profit}</p>
                <p>Profit Percentage: ${auction.profit_percentage}</p>
                <p>Items in AH: ${auction.items_in_ah}</p>
                <p>Item Category: ${auction.item_category}</p>
            `;

            auctionsSection.appendChild(auctionElement);
        });
    }

    // Fetch auction data initially when the page loads
    fetchAuctionData();

    // Set an interval to refresh the data periodically (adjust the interval as needed)
    setInterval(fetchAuctionData, 30000); // Refresh every 30 seconds
});
