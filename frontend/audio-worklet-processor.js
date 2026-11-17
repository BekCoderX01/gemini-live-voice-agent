/**
 * Gemini Live API - 16kHz PCM Audio Worklet Processor
 * Captures microphone input, downsamples to 16,000 Hz mono,
 * converts Float32 to signed 16-bit Little-Endian PCM, and streams chunks.
 */

class PCMRecorderProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetSampleRate = 16000;
    this.downsampleRatio = sampleRate / this.targetSampleRate;
    this.accumulatedSamples = [];
    this.chunkSize = 1024; // ~64ms of 16kHz PCM per chunk
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) {
      return true;
    }

    const channelData = input[0]; // Mono channel Float32Array [-1.0, 1.0]

    // Linear downsampling to targetSampleRate (16kHz)
    if (Math.abs(this.downsampleRatio - 1.0) < 0.01) {
      // AudioContext already running at 16kHz
      for (let i = 0; i < channelData.length; i++) {
        const s = Math.max(-1, Math.min(1, channelData[i]));
        this.accumulatedSamples.push(s < 0 ? s * 0x8000 : s * 0x7FFF);
      }
    } else {
      // Downsample from sampleRate (e.g. 48kHz or 44.1kHz) to 16kHz
      const targetCount = Math.floor(channelData.length / this.downsampleRatio);
      for (let i = 0; i < targetCount; i++) {
        const sourceIdx = Math.floor(i * this.downsampleRatio);
        const s = Math.max(-1, Math.min(1, channelData[sourceIdx]));
        this.accumulatedSamples.push(s < 0 ? s * 0x8000 : s * 0x7FFF);
      }
    }

    // Flush in chunks to maintain low latency (~64ms - 128ms)
    while (this.accumulatedSamples.length >= this.chunkSize) {
      const chunk = this.accumulatedSamples.splice(0, this.chunkSize);
      const int16Array = new Int16Array(chunk);
      // Transfer buffer to avoid copy overhead
      this.port.postMessage(int16Array.buffer, [int16Array.buffer]);
    }

    return true;
  }
}

registerProcessor('pcm-recorder-processor', PCMRecorderProcessor);
