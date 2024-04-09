// the html element of the button on the chrome extension
let reviewScrape = document.getElementById("detect");

// the html element of the review that is copied on the chrome extension
let review_html = document.getElementById('review_data')

// adds the review text to the chrome extension ui
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    let review = request.review;
    
    if(review != null) {
        review_html.innerText = review

    }
    else {
        review_html.innerText = 'No Review Found'
    }

})

// runs the scrapReviewFromPage function when the "detect" button is clicked
reviewScrape.addEventListener("click", async () => {
    let [tab] = await chrome.tabs.query({active: true, currentWindow: true});
    let url = await tab.url;

    chrome.scripting.executeScript({
        args: [url],
        target: {tabId: tab.id},
        func: scrapeReviewFromPage
    })
})

// scrapes the review page
// THIS IS THE MAIN PART YOU NEED TO LOOK AT SANHA
async function scrapeReviewFromPage(url) {
    // the text of the review
    let review = null;
    
    if(url.includes('https://www.amazon.com/gp/customer-reviews/')) {
        rating = document.evaluate("//*[contains(concat(' ', @id, ' '), 'customer_review')]/div[2]/a/i/span", document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue.innerText
        rating = rating[0]
        review = document.evaluate("//*[contains(concat(' ', @id, ' '), 'customer_review')]/div[4]/span/span", document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue.innerText

        
        // here is where you send the post request to the api
        // Make a POST request to your Flask API endpoint
        let data = {
            text: review
        };

        // Make a POST request to your Flask API endpoint with JSON data
        let res = await fetch('http://127.0.0.1:5000/endpoint', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json', // Set the content type to JSON
            },
            body: JSON.stringify(data), // Stringify the JSON object
        })
            .then(response => {
                if (response.ok) {
                    return response.json(); // assuming the server returns JSON
                }
                throw new Error('Network response was not ok.');
            })
            .then(data => {
                console.log(data.text); // Handle the response from the Flask API
                chrome.runtime.sendMessage({ review: data.text });

            })
            .catch(error => {
                console.error('There was a problem with your fetch operation:', error);
            });
            
    }
    
    else {
        chrome.runtime.sendMessage({review});
    }
}