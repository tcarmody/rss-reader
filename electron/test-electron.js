const { app } = require('electron');

console.log('app:', app);
console.log('app.whenReady:', typeof app.whenReady);

app.whenReady().then(() => {
  console.log('Electron is ready!');
  app.quit();
});
