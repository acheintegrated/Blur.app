import './main.css'

// Use Electron's runtime require because nodeIntegration=true
const { ipcRenderer } = window.require('electron')
const fs = window.require('fs').promises
// const path = window.require('path') // if/when needed

console.log('cwd listing:', await fs.readdir('.'))

document.addEventListener('DOMContentLoaded', async () => {
  try {
    const res = await ipcRenderer.invoke('py:run', { args: ['--hello'] })
    console.log('py:', res)
    const status = document.getElementById('status')
    if (status) status.textContent = `py ok · ${res.stdout.slice(0,80)}`
  } catch (e) {
    console.error(e)
  }
})
