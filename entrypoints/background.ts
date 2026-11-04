export default defineBackground(() => {
  console.log('Hello background!', { id: browser.runtime.id });

  browser.runtime.onMessage.addListener(function (message) {
    if (message.type === 'NEW_USER_MESSAGE') {
      console.log('Echo received:', message.text);
    }
  });
});
