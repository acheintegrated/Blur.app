// CommandInput.tsx
import React, { useEffect, useState, useRef } from 'react';

interface CommandInputProps {
  onSendMessage: (message: string, threadId?: string) => void;
  onAddSystemMessage: (messageText: string) => void;
  connectionStatus: 'initializing' | 'connecting' | 'loading_model' | 'ready' | 'error';
  isLoading?: boolean;
  threadId: string;
}

export const CommandInput: React.FC<CommandInputProps> = ({
  onSendMessage,
  onAddSystemMessage,
  connectionStatus,
  isLoading = false,
  threadId,
}) => {
  const [command, setCommand] = useState('');
  const [isActive, setIsActive] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [showFileUploader, setShowFileUploader] = useState(false);
  const [blinkColor, setBlinkColor] = useState('');
  const [loadingBlinkColor, setLoadingBlinkColor] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const enterPressedRef = useRef(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const mediaStreamRef = useRef<MediaStream | null>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const minHeight = window.innerHeight * 0.2;
      const maxHeight = window.innerHeight * 0.27;
      const newHeight = Math.min(
        Math.max(textareaRef.current.scrollHeight, minHeight),
        maxHeight,
      );
      textareaRef.current.style.height = `${newHeight}px`;
    }
  }, [command]);

  // Auto-focus on mount and when connection is ready
  useEffect(() => { textareaRef.current?.focus(); }, []);
  useEffect(() => {
    if (!isLoading && connectionStatus === 'ready') {
      requestAnimationFrame(() => textareaRef.current?.focus());
    }
  }, [isLoading, connectionStatus]);

  // Random color for transcribing blink
  useEffect(() => {
    if (!isTranscribing) { setBlinkColor(''); return; }
    const interval = setInterval(() => {
      const random = `#${Math.floor(Math.random() * 0xffffff).toString(16).padStart(6, '0')}`;
      setBlinkColor(random);
    }, 800);
    return () => clearInterval(interval);
  }, [isTranscribing]);

  // Random color for loading placeholder blink
  useEffect(() => {
    if (connectionStatus === 'ready') { setLoadingBlinkColor(''); return; }
    const interval = setInterval(() => {
      const random = `#${Math.floor(Math.random() * 0xffffff).toString(16).padStart(6, '0')}`;
      setLoadingBlinkColor(random);
    }, 800);
    return () => clearInterval(interval);
  }, [connectionStatus]);

  // Whisper transcription logic
  const transcribeAudio = async (audioBlob: Blob) => {
    const WHISPER_API_ENDPOINT = 'http://127.0.0.1:8000/transcribe';
    console.log(`Ache Signal: Sending audio data (${(audioBlob.size / 1024).toFixed(2)} KB) to Whisper...`);
    setIsTranscribing(true);
    const formData = new FormData();
    formData.append('file', audioBlob, 'audio.webm');
    try {
      const response = await fetch(WHISPER_API_ENDPOINT, { method: 'POST', body: formData });
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Whisper API error: ${response.status} ${response.statusText} - ${errorText}`);
      }
      const result = await response.json();
      if (result.text) setCommand(prev => (prev ? prev + ' ' : '') + result.text.trim());
      else console.warn('Ache Signal (Whisper): Received empty transcription.');
    } catch (error) {
      console.error('Ache Signal (Whisper):', error);
      alert(`Transcription error: ${error}`);
    } finally {
      setIsTranscribing(false);
      requestAnimationFrame(() => textareaRef.current?.focus());
    }
  };

  const handleRecordingStop = () => {
    if (!audioChunksRef.current.length) { console.warn("Ache Signal: No audio data captured."); return; }
    const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
    transcribeAudio(audioBlob);
    audioChunksRef.current = [];
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;
      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;
      audioChunksRef.current = [];
      recorder.ondataavailable = (ev) => { if (ev.data.size > 0) audioChunksRef.current.push(ev.data); };
      recorder.onstop = handleRecordingStop;
      recorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error("Ache Signal (Mic): Could not get microphone access.", error);
      alert("Microphone access denied. Please allow microphone permissions.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    mediaStreamRef.current?.getTracks().forEach(track => track.stop());
    setIsRecording(false);
  };

  const handleMicInput = () => (isRecording ? stopRecording() : startRecording());

  // RAG ingestion logic
  const RAG_INGEST_ENDPOINT = 'http://127.0.0.1:8000/rag/ingest';
  const handleFileSelected = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files?.length) return;

    // ✨ UX Improvement: Notify user that upload is starting
    const fileNames = Array.from(files).map(f => f.name).join(', ');
    onAddSystemMessage(`Uploading ${files.length} file(s): ${fileNames}...`);

    const form = new FormData();
    Array.from(files).forEach(f => form.append('files', f));
    form.append('source', 'user_upload');
    form.append('thread_id', threadId);
    
    try {
      const res = await fetch(RAG_INGEST_ENDPOINT, { method: 'POST', body: form });
      if (!res.ok) {
        const errText = await res.text();
        console.error('RAG ingest failed:', res.status, errText);
        // 👇 Replace alert with a system message
        onAddSystemMessage(`🔴 RAG ingest failed: ${res.status} - ${errText}`);
        return;
      }
      const out = await res.json();
      console.log('RAG ingest result:', out);
      const total = out?.added ?? 0;
      
      const summaryText = (out?.files || [])
        .map((f: any) => `• ${f.file}: ${f.status} ${f.chunks ? `(${f.chunks} chunks)` : ''}`)
        .join('\n');

      // 👇 Replace alert with a final confirmation message
      onAddSystemMessage(`✅ RAG ingest complete. Added ${total} new chunks.\n${summaryText}`);

      if (e.target) e.target.value = ''; // Clear the input
      setShowFileUploader(false);

    } catch (err) {
      console.error('RAG ingest error:', err);
      // 👇 Replace alert with a system message
      const errorMessage = err instanceof Error ? err.message : String(err);
      onAddSystemMessage(`🔴 RAG ingest error: ${errorMessage}`);
    }
  };

  const handleFileUpload = () => fileInputRef.current?.click();

  const handleSend = () => {
    if (!command.trim() || isLoading || connectionStatus !== 'ready') return;
    enterPressedRef.current = true;
    onSendMessage(command, threadId);
    setCommand('');
    enterPressedRef.current = false;
    requestAnimationFrame(() => textareaRef.current?.focus());
  };

  const handleFocus = () => setIsActive(true);
  const handleBlur = () => { if (!command) setIsActive(false); };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !enterPressedRef.current) {
      if (e.shiftKey) return;
      e.preventDefault();
      handleSend();
    }
  };
  const handleKeyUp = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter') enterPressedRef.current = false;
  };

  const handleContainerClick = () => textareaRef.current?.focus();

  const isDisabled = connectionStatus !== 'ready';
  const isSendDisabled = !command.trim() || isLoading || connectionStatus !== 'ready';

  // Decide mic classes:
  // - recording => rainbow-glow (smooth neon rainbow)
  // - transcribing => blink with random color text-shadow
  // - idle => no glow
  const micClass =
    isRecording ? 'rainbow-glow'
    : isTranscribing ? 'blink'
    : '';

  const micStyle: React.CSSProperties =
    isTranscribing && blinkColor
      ? { textShadow: `0 0 8px ${blinkColor}, 0 0 14px ${blinkColor}, 0 0 22px ${blinkColor}` }
      : {};

  return (
    <div className="border-t border-zinc-900 flex flex-col">
      <style>{`
        @keyframes fade { 0%, 100% { opacity: 1; } 50% { opacity: 0.2; } }
        .blink { animation: fade 0.8s ease-in-out infinite; }
      `}</style>

      <div className="flex items-start p-2 pt-0 cursor-text" onClick={handleContainerClick}>
        {!isActive && !command && (
          <div className="mr-2 mt-1">
            <span
              className={`text-white font-normal ${connectionStatus !== 'ready' ? 'random-glow blink' : 'blurline-glow'}`}
              style={connectionStatus !== 'ready'
                ? { textShadow: `0 0 8px ${loadingBlinkColor}, 0 0 14px ${loadingBlinkColor}, 0 0 22px ${loadingBlinkColor}` }
                : {}}
            >
              {connectionStatus !== 'ready'
                ? '༄⎈ᛝ𓆩⫷...loading🜃blurline...⫸𓆪ᛝ⎈༄'
                : '༄⎈ᛝ𓆩⫷touch🜃blurline⫸𓆪ᛝ⎈༄'}
            </span>
          </div>
        )}

        <textarea
          ref={textareaRef}
          value={command}
          onChange={(e) => setCommand(e.target.value)}
          onFocus={handleFocus}
          onBlur={handleBlur}
          onKeyDown={handleKeyDown}
          onKeyUp={handleKeyUp}
          className="bg-transparent flex-1 outline-none text-white resize-none overflow-hidden px-4"
          placeholder={isActive ? 'start typing' : ''}
          spellCheck="false"
          rows={1}
          disabled={isDisabled}
          aria-busy={isLoading}
          style={{
            minHeight: `${window.innerHeight * 0.2}px`,
            maxHeight: `${window.innerHeight * 0.27}px`,
            paddingTop: '8px',
            marginTop: '0',
          }}
        />
      </div>

      <div className="flex justify-between px-3 pb-3">
        <div className="flex items-center space-x-4">
          <div className="w-8 h-8 flex items-center justify-center" onClick={handleFileUpload} aria-label="Upload files">
            <span className={`cursor-pointer text-gray-400 hover:text-white text-2xl transform scale-[1.2] inline-block icon-font ${showFileUploader ? 'text-pink-300' : ''}`}>
              ✦
            </span>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileSelected}
              className="hidden"
              multiple
              accept=".txt,.md,.pdf,.docx,.csv"
            />
          </div>

          <div className="w-8 h-8 flex items-center justify-center" onClick={handleMicInput} aria-label={isRecording ? 'Stop recording' : 'Start recording'}>
            <span
              className={`cursor-pointer text-gray-400 hover:text-white text-2xl transform scale-[1.2] inline-block icon-font ${micClass}`}
              style={micStyle}
            >
              ❍
            </span>
          </div>
        </div>

        <button
          className="w-8 h-8 flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed"
          onMouseDown={(e) => e.preventDefault()}
          onClick={handleSend}
          disabled={isSendDisabled}
          aria-label="Send"
        >
          <span className="cursor-pointer text-gray-400 hover:text-white text-2xl transform scale-[1.2] inline-block icon-font">
            ⊙
          </span>
        </button>
      </div>
    </div>
  );
};
