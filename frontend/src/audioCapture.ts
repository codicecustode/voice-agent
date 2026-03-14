/**
 * audioCapture.ts — Capture microphone audio and stream it to the backend.
 *
 * We use the Web Audio API to:
 *   1. Get microphone access
 *   2. Downsample audio to 16kHz PCM (what Deepgram expects)
 *   3. Send raw bytes over WebSocket
 */

export class AudioCapture {
  private context: AudioContext | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private processor: ScriptProcessorNode | null = null;
  private stream: MediaStream | null = null;
  private onChunk: (data: ArrayBuffer) => void;

  constructor(onChunk: (data: ArrayBuffer) => void) {
    this.onChunk = onChunk;
  }

  async start(): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: 16000,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    this.context = new AudioContext({ sampleRate: 16000 });
    this.source = this.context.createMediaStreamSource(this.stream);

    // ScriptProcessorNode gives us raw PCM access
    // Buffer size 4096 = ~256ms of audio at 16kHz
    this.processor = this.context.createScriptProcessor(4096, 1, 1);

    this.processor.onaudioprocess = (event) => {
      const inputData = event.inputBuffer.getChannelData(0);
      // Convert Float32 [-1, 1] to Int16 PCM
      const pcm = this.float32ToInt16(inputData);
      this.onChunk(pcm.buffer);
    };

    this.source.connect(this.processor);
    this.processor.connect(this.context.destination);
  }

  stop(): void {
    this.processor?.disconnect();
    this.source?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    this.context?.close();
  }

  private float32ToInt16(buffer: Float32Array): Int16Array {
    const result = new Int16Array(buffer.length);
    for (let i = 0; i < buffer.length; i++) {
      // Clamp and convert
      const s = Math.max(-1, Math.min(1, buffer[i]));
      result[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return result;
  }
}
