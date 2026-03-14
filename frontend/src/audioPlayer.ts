/**
 * audioPlayer.ts — Play streaming PCM audio from the backend.
 *
 * Receives raw PCM chunks from the WebSocket and plays them
 * in the correct order with minimal buffering.
 *
 * Barge-in: when the backend signals "barge_in", we immediately
 * stop playback and clear the queue so the agent can respond
 * to the new input.
 */

export class AudioPlayer {
  private context: AudioContext;
  private queue: AudioBuffer[] = [];
  private isPlaying = false;
  private nextStartTime = 0;
  private currentSource: AudioBufferSourceNode | null = null;
  private readonly SAMPLE_RATE = 16000;

  constructor() {
    this.context = new AudioContext({ sampleRate: this.SAMPLE_RATE });
  }

  /**
   * Add a chunk of raw Int16 PCM to the playback queue.
   * Call this each time a WebSocket binary message arrives.
   */
  enqueue(pcmBytes: ArrayBuffer): void {
    const int16 = new Int16Array(pcmBytes);
    const float32 = new Float32Array(int16.length);

    // Convert Int16 to Float32 for Web Audio API
    for (let i = 0; i < int16.length; i++) {
      float32[i] = int16[i] / (int16[i] < 0 ? 0x8000 : 0x7fff);
    }

    const buffer = this.context.createBuffer(1, float32.length, this.SAMPLE_RATE);
    buffer.copyToChannel(float32, 0);
    this.queue.push(buffer);

    if (!this.isPlaying) {
      this._playNext();
    }
  }

  /**
   * Stop all playback immediately and clear the queue.
   * Called when barge-in is detected.
   */
  stop(): void {
    this.queue = [];
    this.isPlaying = false;
    if (this.currentSource) {
      try {
        this.currentSource.stop();
      } catch {
        // Already stopped
      }
      this.currentSource = null;
    }
    this.nextStartTime = 0;
  }

  private _playNext(): void {
    if (this.queue.length === 0) {
      this.isPlaying = false;
      return;
    }

    this.isPlaying = true;
    const buffer = this.queue.shift()!;
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.context.destination);

    const now = this.context.currentTime;
    const startAt = Math.max(now, this.nextStartTime);
    source.start(startAt);
    this.nextStartTime = startAt + buffer.duration;

    this.currentSource = source;
    source.onended = () => {
      this._playNext();
    };
  }
}
