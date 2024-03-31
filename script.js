// the html element of the button on the chrome extension
let reviewScrape = document.getElementById("detect");

// the html element of the review that is copied on the chrome extension
let body = document.getElementById('review_data')

// adds the review text to the chrome extension ui
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    let review = request.review;
    
    if(review != null) {
        body.innerText = review
    }
})

// runs the scrapReviewFromPage function when the "detect" button is clicked
reviewScrape.addEventListener("click", async () => {
    let [tab] = await chrome.tabs.query({active: true, currentWindow: true});

    chrome.scripting.executeScript({
        target: {tabId: tab.id},
        func: scrapeReviewFromPage,
    })
})

// scrapes the review page
// THIS IS THE MAIN PART YOU NEED TO LOOK AT SANHA
async function scrapeReviewFromPage() {
    // the text of the review
    let review = document.evaluate("//*[contains(concat(' ', @id, ' '), 'customer_review')]/div[4]/span/span", document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue.innerText
    
    // here is where you send the post request to the api

    chrome.runtime.sendMessage({review});
}