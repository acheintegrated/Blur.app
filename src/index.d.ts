// Type definitions for global objects
declare global {
  interface Window {
    SpeechRecognition: any
    webkitSpeechRecognition: any
    recognition?: any // For storing speech recognition instance
  }
}
export {}
