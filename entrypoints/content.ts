export default defineContentScript({
  matches: ['https://chatgpt.com/*'],

  main() {
    console.log('Echo: ChatGPT content script loaded.');
  },
});