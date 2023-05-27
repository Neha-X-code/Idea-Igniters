// background.js
chrome.tabs.onUpdated.addListener(function (tabId, changeInfo, tab) {
    if (changeInfo.status === 'complete') {
      if (tab.url.includes('example.com')) {
        chrome.scripting.executeScript({
          target: { tabId: tabId },
          function: showAlert
        });
      }
    }
  });
  
  function showAlert() {
    alert('Warning: This website may contain misleading medicine articles.');
  }
  