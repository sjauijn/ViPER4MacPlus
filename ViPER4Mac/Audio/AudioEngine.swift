import Accelerate
import AppKit
import AudioToolbox
import CoreAudio
import Foundation

private let logger = AppLogger(category: "AudioEngine")

private enum TapSource: Equatable {
  case everythingExceptSelf
  case fineTune(AudioObjectID)
}

private let fineTuneBundleID = "com.finetuneapp.FineTune"
private let fineTuneFallbackDelay: TimeInterval = 3.0
private let tapSourceCoalesceDelay: TimeInterval = 0.15
private let tapSourceSafetyNetInterval: TimeInterval = 10.0

final class AudioEngine {
  static let shared = AudioEngine()

  let viperBridge = ViPERBridge()

  private(set) var outputDeviceID: AudioDeviceID = kAudioObjectUnknown
  private let channelCount: UInt32 = 2

  private var isRunning = false
  private var deviceListenerInstalled = false
  private var deviceListChangedListenerInstalled = false
  var onOutputDeviceChanged: (() -> Void)?

  var processingEnabled = true

  private var tapID: AudioObjectID = kAudioObjectUnknown
  private var aggregateDeviceID: AudioObjectID = kAudioObjectUnknown
  private var ioProcID: AudioDeviceIOProcID?
  private var tapUUID: NSUUID?

  private var callbackCount: UInt64 = 0

  private var tapSource: TapSource = .everythingExceptSelf
  private var processListListenerBlock: AudioObjectPropertyListenerBlock?
  private var fineTuneProcessListener: (objectID: AudioObjectID, block: AudioObjectPropertyListenerBlock)?
  private var workspaceObservers: [NSObjectProtocol] = []
  private var safetyNetTimer: Timer?
  private var evaluationWork: DispatchWorkItem?
  private var fallbackWork: DispatchWorkItem?
  private var processListAddress = AudioObjectPropertyAddress(
    mSelector: kAudioHardwarePropertyProcessObjectList,
    mScope: kAudioObjectPropertyScopeGlobal,
    mElement: kAudioObjectPropertyElementMain
  )

  private init() {}

  deinit {
    stop()
  }

  // MARK: - Public Interface

  func start() {
    guard !isRunning else { return }

    guard let realDevice = findRealOutputDevice() else {
      logger.error("No real output device found.")
      return
    }
    outputDeviceID = realDevice

    let sampleRate = getSampleRate(for: outputDeviceID)
    viperBridge.setSamplingRate(UInt32(sampleRate))
    logger.info("AudioEngine: output=\(realDevice) rate=\(sampleRate)")

    guard createTap() else {
      logger.error("Failed to create process tap")
      return
    }

    guard createAggregateDevice() else {
      logger.error("Failed to create aggregate device")
      destroyTap()
      return
    }

    guard createIOProc() else {
      logger.error("Failed to create IOProc")
      destroyAggregateDevice()
      destroyTap()
      return
    }

    let status = AudioDeviceStart(aggregateDeviceID, ioProcID!)
    if status != noErr {
      logger.error("AudioDeviceStart failed: \(status)")
      destroyIOProc()
      destroyAggregateDevice()
      destroyTap()
      return
    }

    isRunning = true
    installDeviceListListener()
    startTapSourceMonitor()
    logger.info(
      "Audio engine started. Output=\(outputDeviceID) DSP=\(processingEnabled)"
    )
  }

  func stop() {
    guard isRunning else { return }

    stopTapSourceMonitor()
    removeDeviceListListener()
    teardownPipeline()

    isRunning = false
    callbackCount = 0
    logger.info("Audio engine stopped")
  }

  var outputDeviceName: String {
    guard outputDeviceID != kAudioObjectUnknown else { return "None" }
    return getDeviceName(outputDeviceID)
  }

  // MARK: - Process Tap

  private func getCurrentProcessObjectID() -> AudioObjectID? {
    var addr = AudioObjectPropertyAddress(
      mSelector: kAudioHardwarePropertyTranslatePIDToProcessObject,
      mScope: kAudioObjectPropertyScopeGlobal,
      mElement: kAudioObjectPropertyElementMain
    )
    var pid = pid_t(ProcessInfo.processInfo.processIdentifier)
    var processObjectID: AudioObjectID = kAudioObjectUnknown
    var size = UInt32(MemoryLayout<AudioObjectID>.size)
    let status = AudioObjectGetPropertyData(
      AudioObjectID(kAudioObjectSystemObject),
      &addr,
      UInt32(MemoryLayout<pid_t>.size),
      &pid,
      &size,
      &processObjectID
    )
    if status != noErr {
      logger.error("TranslatePIDToProcessObject failed: \(status)")
      return nil
    }
    return processObjectID
  }

  private func processObjectID(forPID pid: pid_t) -> AudioObjectID? {
    var addr = AudioObjectPropertyAddress(
      mSelector: kAudioHardwarePropertyTranslatePIDToProcessObject,
      mScope: kAudioObjectPropertyScopeGlobal,
      mElement: kAudioObjectPropertyElementMain
    )
    var inPID = pid
    var objectID: AudioObjectID = kAudioObjectUnknown
    var size = UInt32(MemoryLayout<AudioObjectID>.size)
    let status = AudioObjectGetPropertyData(
      AudioObjectID(kAudioObjectSystemObject),
      &addr,
      UInt32(MemoryLayout<pid_t>.size),
      &inPID,
      &size,
      &objectID
    )
    guard status == noErr, objectID != kAudioObjectUnknown else { return nil }
    return objectID
  }

  private func isProcessRunning(_ objectID: AudioObjectID) -> Bool {
    var addr = AudioObjectPropertyAddress(
      mSelector: kAudioProcessPropertyIsRunning,
      mScope: kAudioObjectPropertyScopeGlobal,
      mElement: kAudioObjectPropertyElementMain
    )
    var value: UInt32 = 0
    var size = UInt32(MemoryLayout<UInt32>.size)
    let status = AudioObjectGetPropertyData(objectID, &addr, 0, nil, &size, &value)
    return status == noErr && value != 0
  }

  private func findFineTuneProcessObject() -> AudioObjectID? {
    for app in NSRunningApplication.runningApplications(withBundleIdentifier: fineTuneBundleID) {
      if let objectID = processObjectID(forPID: app.processIdentifier) {
        return objectID
      }
    }
    return nil
  }

  private func resolveTapSource() -> TapSource {
    if let objectID = findFineTuneProcessObject(), isProcessRunning(objectID) {
      return .fineTune(objectID)
    }
    return .everythingExceptSelf
  }

  private func createTap() -> Bool {
    let source = resolveTapSource()

    let desc: CATapDescription
    switch source {
    case let .fineTune(fineTuneObjectID):
      desc = CATapDescription(stereoMixdownOfProcesses: [fineTuneObjectID])
    case .everythingExceptSelf:
      guard let processObjID = getCurrentProcessObjectID() else {
        logger.error("Could not get process object ID for current process")
        return false
      }
      desc = CATapDescription(
        __stereoGlobalTapButExcludeProcesses: [NSNumber(value: processObjID)]
      )
    }
    desc.muteBehavior = .muted
    desc.isPrivate = true
    desc.name = "ViPER4Mac Tap"

    tapUUID = desc.uuid as NSUUID

    var newTapID: AudioObjectID = kAudioObjectUnknown
    let status = AudioHardwareCreateProcessTap(desc, &newTapID)
    if status != noErr {
      logger.error("AudioHardwareCreateProcessTap failed: \(status)")
      return false
    }
    tapID = newTapID
    tapSource = source
    logger.info("Process tap created: \(tapID) uuid=\(desc.uuid.uuidString) source=\(source)")
    return true
  }

  private func destroyTap() {
    guard tapID != kAudioObjectUnknown else { return }
    let status = AudioHardwareDestroyProcessTap(tapID)
    if status != noErr {
      logger.error("AudioHardwareDestroyProcessTap failed: \(status)")
    }
    tapID = kAudioObjectUnknown
    tapUUID = nil
  }

  // MARK: - Aggregate Device

  private func createAggregateDevice() -> Bool {
    guard let uuid = tapUUID else {
      logger.error("No tap UUID available")
      return false
    }

    let outputUID = getDeviceUID(outputDeviceID)
    guard !outputUID.isEmpty else {
      logger.error("Could not get UID for output device \(outputDeviceID)")
      return false
    }

    let tapDict: [String: Any] = [
      kAudioSubTapUIDKey: uuid.uuidString,
      kAudioSubTapDriftCompensationKey: 1,
    ]

    let subDeviceDict: [String: Any] = [
      kAudioSubDeviceUIDKey: outputUID,
    ]

    let aggDesc: [String: Any] = [
      kAudioAggregateDeviceNameKey: "ViPER4Mac Aggregate",
      kAudioAggregateDeviceUIDKey: "com.viper4mac.aggregate.\(UUID().uuidString)",
      kAudioAggregateDeviceMainSubDeviceKey: outputUID,
      kAudioAggregateDeviceIsPrivateKey: true,
      kAudioAggregateDeviceTapAutoStartKey: true,
      kAudioAggregateDeviceTapListKey: [tapDict],
      kAudioAggregateDeviceSubDeviceListKey: [subDeviceDict],
    ]

    var newAggID: AudioObjectID = kAudioObjectUnknown
    let status = AudioHardwareCreateAggregateDevice(aggDesc as CFDictionary, &newAggID)
    if status != noErr {
      logger.error("AudioHardwareCreateAggregateDevice failed: \(status)")
      return false
    }
    aggregateDeviceID = newAggID
    logger.info("Aggregate device created: \(aggregateDeviceID) mainSub=\(outputUID)")
    return true
  }

  private func destroyAggregateDevice() {
    guard aggregateDeviceID != kAudioObjectUnknown else { return }
    let status = AudioHardwareDestroyAggregateDevice(aggregateDeviceID)
    if status != noErr {
      logger.error("AudioHardwareDestroyAggregateDevice failed: \(status)")
    }
    aggregateDeviceID = kAudioObjectUnknown
  }

  // MARK: - IOProc

  private func createIOProc() -> Bool {
    var procID: AudioDeviceIOProcID?

    let ch = Int(channelCount)

    let status = AudioDeviceCreateIOProcIDWithBlock(
      &procID,
      aggregateDeviceID,
      nil
    ) { [weak self] _, inInputData, _, outOutputData, _ in
      guard let self else { return }

      self.callbackCount += 1

      let inABL = UnsafeMutableAudioBufferListPointer(
        UnsafeMutablePointer(mutating: inInputData)
      )
      let outABL = UnsafeMutableAudioBufferListPointer(outOutputData)

      // The device's capture stream comes before the process tap.
      // `.first` could read the silent mic instead of the tap, causing
      // silence when the tap is muted. The tap is always the last stream.
      guard let inBuf = inABL.last,
            let outBuf = outABL.first,
            let inData = inBuf.mData,
            let outData = outBuf.mData
      else { return }

      let outBytes = Int(outBuf.mDataByteSize)
      let inBytes = Int(inBuf.mDataByteSize)
      let copyBytes = min(inBytes, outBytes)

      memcpy(outData, inData, copyBytes)

      let frameCount = copyBytes / (MemoryLayout<Float>.size * ch)
      let sampleCount = frameCount * ch
      let floatPtr = outData.assumingMemoryBound(to: Float.self)

      var maxSample: Float = 0.0
      vDSP_maxmgv(floatPtr, 1, &maxSample, vDSP_Length(sampleCount))

      let hasAudio = maxSample > 1e-6
      if self.processingEnabled && hasAudio {
        self.viperBridge.processAudio(floatPtr, frameCount: UInt32(frameCount))
      }

      if self.callbackCount % 5000 == 1 {
        let fillLevel = inBytes > 0 ? (copyBytes * 100 / inBytes) : 0
        logger.debug(
          "IOPROC: frames=\(frameCount) fill=\(fillLevel)% max=\(maxSample)"
        )
      }
    }

    if status != noErr {
      logger.error("AudioDeviceCreateIOProcIDWithBlock failed: \(status)")
      return false
    }
    ioProcID = procID
    return true
  }

  private func destroyIOProc() {
    guard let procID = ioProcID else { return }
    AudioDeviceDestroyIOProcID(aggregateDeviceID, procID)
    ioProcID = nil
  }

  // MARK: - Pipeline Teardown

  private func teardownPipeline() {
    if let procID = ioProcID {
      AudioDeviceStop(aggregateDeviceID, procID)
    }
    destroyIOProc()
    destroyAggregateDevice()
    destroyTap()
  }

  private func startTapSourceMonitor() {
    guard processListListenerBlock == nil else { return }

    let listBlock: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
      self?.scheduleTapSourceEvaluation()
    }
    let status = AudioObjectAddPropertyListenerBlock(
      AudioObjectID(kAudioObjectSystemObject),
      &processListAddress,
      .main,
      listBlock
    )
    if status == noErr {
      processListListenerBlock = listBlock
    } else {
      logger.error("Failed to install process list listener: \(status)")
    }

    let center = NSWorkspace.shared.notificationCenter
    let names = [
      NSWorkspace.didLaunchApplicationNotification,
      NSWorkspace.didTerminateApplicationNotification,
    ]
    for name in names {
      let token = center.addObserver(forName: name, object: nil, queue: .main) { [weak self] note in
        let app = note.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication
        guard app?.bundleIdentifier == fineTuneBundleID else { return }
        self?.scheduleTapSourceEvaluation()
      }
      workspaceObservers.append(token)
    }

    let timer = Timer(timeInterval: tapSourceSafetyNetInterval, repeats: true) { [weak self] _ in
      self?.scheduleTapSourceEvaluation()
    }
    timer.tolerance = tapSourceSafetyNetInterval * 0.2
    RunLoop.main.add(timer, forMode: .common)
    safetyNetTimer = timer

    scheduleTapSourceEvaluation()
  }

  private func stopTapSourceMonitor() {
    if let block = processListListenerBlock {
      AudioObjectRemovePropertyListenerBlock(
        AudioObjectID(kAudioObjectSystemObject),
        &processListAddress,
        .main,
        block
      )
      processListListenerBlock = nil
    }

    detachFineTuneProcessListener()

    for token in workspaceObservers {
      NSWorkspace.shared.notificationCenter.removeObserver(token)
    }
    workspaceObservers.removeAll()

    safetyNetTimer?.invalidate()
    safetyNetTimer = nil

    evaluationWork?.cancel()
    evaluationWork = nil
    cancelFallback()
  }

  private func attachFineTuneProcessListener(to objectID: AudioObjectID) {
    var addr = AudioObjectPropertyAddress(
      mSelector: kAudioProcessPropertyIsRunning,
      mScope: kAudioObjectPropertyScopeGlobal,
      mElement: kAudioObjectPropertyElementMain
    )
    let block: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
      self?.scheduleTapSourceEvaluation()
    }
    let status = AudioObjectAddPropertyListenerBlock(objectID, &addr, .main, block)
    if status == noErr {
      fineTuneProcessListener = (objectID, block)
    } else {
      logger.warning("Failed to install FineTune process listener: \(status)")
    }
  }

  private func detachFineTuneProcessListener() {
    guard let listener = fineTuneProcessListener else { return }
    var addr = AudioObjectPropertyAddress(
      mSelector: kAudioProcessPropertyIsRunning,
      mScope: kAudioObjectPropertyScopeGlobal,
      mElement: kAudioObjectPropertyElementMain
    )
    let status = AudioObjectRemovePropertyListenerBlock(
      listener.objectID, &addr, .main, listener.block
    )
    if status != noErr, status != OSStatus(kAudioHardwareBadObjectError) {
      logger.warning("Failed to remove FineTune process listener: \(status)")
    }
    fineTuneProcessListener = nil
  }

  private func syncFineTuneProcessListener() {
    let current = findFineTuneProcessObject()
    if fineTuneProcessListener?.objectID == current { return }
    detachFineTuneProcessListener()
    if let current {
      attachFineTuneProcessListener(to: current)
    }
  }

  private func scheduleTapSourceEvaluation() {
    evaluationWork?.cancel()
    let work = DispatchWorkItem { [weak self] in
      self?.evaluateTapSource()
    }
    evaluationWork = work
    DispatchQueue.main.asyncAfter(deadline: .now() + tapSourceCoalesceDelay, execute: work)
  }

  private func cancelFallback() {
    fallbackWork?.cancel()
    fallbackWork = nil
  }

  private func evaluateTapSource() {
    guard isRunning else { return }

    syncFineTuneProcessListener()

    let desired = resolveTapSource()
    if desired == tapSource {
      cancelFallback()
      return
    }

    if case .fineTune = tapSource, desired == .everythingExceptSelf {
      guard fallbackWork == nil else { return }
      let work = DispatchWorkItem { [weak self] in
        self?.fallbackWork = nil
        self?.applyResolvedTapSource()
      }
      fallbackWork = work
      DispatchQueue.main.asyncAfter(deadline: .now() + fineTuneFallbackDelay, execute: work)
      return
    }

    cancelFallback()
    applyResolvedTapSource()
  }

  private func applyResolvedTapSource() {
    guard isRunning else { return }
    let desired = resolveTapSource()
    guard desired != tapSource else { return }
    logger.info("Tap source changing: \(tapSource) -> \(desired)")
    rebuildPipeline()
  }

  private func rebuildPipeline() {
    teardownPipeline()

    guard createTap() else {
      logger.error("Failed to create tap while rebuilding pipeline")
      return
    }

    guard createAggregateDevice() else {
      logger.error("Failed to create aggregate device while rebuilding pipeline")
      destroyTap()
      return
    }

    guard createIOProc() else {
      logger.error("Failed to create IOProc while rebuilding pipeline")
      destroyAggregateDevice()
      destroyTap()
      return
    }

    let status = AudioDeviceStart(aggregateDeviceID, ioProcID!)
    if status != noErr {
      logger.error("Failed to start aggregate device while rebuilding pipeline: \(status)")
      destroyIOProc()
      destroyAggregateDevice()
      destroyTap()
      return
    }

    callbackCount = 0
  }

  // MARK: - Device Change Handling

  private func installDeviceListListener() {
    let selfPtr = Unmanaged.passUnretained(self).toOpaque()

    if !deviceListenerInstalled {
      var addr = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDefaultOutputDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
      )
      let status = AudioObjectAddPropertyListener(
        AudioObjectID(kAudioObjectSystemObject),
        &addr,
        engineDeviceChangedCallback,
        selfPtr
      )
      if status == noErr {
        deviceListenerInstalled = true
      } else {
        logger.error("Failed to install default device listener: \(status)")
      }
    }

    if !deviceListChangedListenerInstalled {
      var addr = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDevices,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
      )
      let status = AudioObjectAddPropertyListener(
        AudioObjectID(kAudioObjectSystemObject),
        &addr,
        engineDeviceListChangedCallback,
        selfPtr
      )
      if status == noErr {
        deviceListChangedListenerInstalled = true
      } else {
        logger.error("Failed to install device list listener: \(status)")
      }
    }

    logger.info("Engine device listeners installed")
  }

  private func removeDeviceListListener() {
    let selfPtr = Unmanaged.passUnretained(self).toOpaque()

    if deviceListenerInstalled {
      var addr = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDefaultOutputDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
      )
      AudioObjectRemovePropertyListener(
        AudioObjectID(kAudioObjectSystemObject),
        &addr,
        engineDeviceChangedCallback,
        selfPtr
      )
      deviceListenerInstalled = false
    }

    if deviceListChangedListenerInstalled {
      var addr = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDevices,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
      )
      AudioObjectRemovePropertyListener(
        AudioObjectID(kAudioObjectSystemObject),
        &addr,
        engineDeviceListChangedCallback,
        selfPtr
      )
      deviceListChangedListenerInstalled = false
    }
  }

  func handleDefaultDeviceChanged() {
    guard isRunning else { return }
    let newDefault = getDefaultOutputDevice()

    if newDefault == outputDeviceID { return }
    if !hasOutputStreams(newDefault) { return }
    if !isDeviceAlive(newDefault) {
      logger.info("Ignoring change to dead device: \(getDeviceName(newDefault))")
      return
    }
    switchOutputDevice(to: newDefault)
  }

  func handleDeviceListChanged() {
    guard isRunning else { return }

    let allDevices = getAllDeviceIDs()
    let outputStillExists =
      allDevices.contains(outputDeviceID)
        && hasOutputStreams(outputDeviceID)
        && isDeviceAlive(outputDeviceID)

    if outputStillExists {
      let newDefault = getDefaultOutputDevice()
      if newDefault != outputDeviceID
        && hasOutputStreams(newDefault)
        && isDeviceAlive(newDefault)
      {
        switchOutputDevice(to: newDefault)
      }
      return
    }

    guard let fallback = findRealOutputDevice() else {
      logger.error("Device disappeared but no fallback output found")
      return
    }

    logger.info(
      "Output device disappeared: \(getDeviceName(outputDeviceID)) -> \(getDeviceName(fallback))"
    )
    switchOutputDevice(to: fallback)
  }

  func switchOutputDevice(to newDevice: AudioDeviceID) {
    let oldName = getDeviceName(outputDeviceID)
    let newName = getDeviceName(newDevice)
    logger.info("Switching output: \(oldName) -> \(newName)")

    teardownPipeline()

    outputDeviceID = newDevice

    let sampleRate = getSampleRate(for: outputDeviceID)
    viperBridge.setSamplingRate(UInt32(sampleRate))

    guard createTap() else {
      logger.error("Failed to create tap for new device")
      return
    }

    guard createAggregateDevice() else {
      logger.error("Failed to create aggregate device for new device")
      destroyTap()
      return
    }

    guard createIOProc() else {
      logger.error("Failed to create IOProc for new device")
      destroyAggregateDevice()
      destroyTap()
      return
    }

    let status = AudioDeviceStart(aggregateDeviceID, ioProcID!)
    if status != noErr {
      logger.error("Failed to start aggregate device for new device: \(status)")
      destroyIOProc()
      destroyAggregateDevice()
      destroyTap()
      return
    }

    callbackCount = 0
    onOutputDeviceChanged?()
    logger.info("Output re-routed to \(newName) rate=\(sampleRate)")
  }

  // MARK: - Device Discovery

  private func findRealOutputDevice() -> AudioDeviceID? {
    let currentDefault = getDefaultOutputDevice()
    if hasOutputStreams(currentDefault)
      && isDeviceAlive(currentDefault)
    {
      return currentDefault
    }

    let deviceIDs = getAllDeviceIDs()
    for deviceID in deviceIDs {
      if hasOutputStreams(deviceID) && isDeviceAlive(deviceID) {
        return deviceID
      }
    }
    return nil
  }

  private func getAllDeviceIDs() -> [AudioDeviceID] {
    var propAddr = AudioObjectPropertyAddress(
      mSelector: kAudioHardwarePropertyDevices,
      mScope: kAudioObjectPropertyScopeGlobal,
      mElement: kAudioObjectPropertyElementMain
    )
    var dataSize: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(
      AudioObjectID(kAudioObjectSystemObject), &propAddr, 0, nil, &dataSize
    ) == noErr
    else { return [] }

    let count = Int(dataSize) / MemoryLayout<AudioDeviceID>.size
    var deviceIDs = [AudioDeviceID](repeating: 0, count: count)
    guard AudioObjectGetPropertyData(
      AudioObjectID(kAudioObjectSystemObject), &propAddr, 0, nil, &dataSize, &deviceIDs
    ) == noErr
    else { return [] }

    return deviceIDs
  }

  private func getDeviceUID(_ deviceID: AudioDeviceID) -> String {
    var propAddr = AudioObjectPropertyAddress(
      mSelector: kAudioDevicePropertyDeviceUID,
      mScope: kAudioObjectPropertyScopeGlobal,
      mElement: kAudioObjectPropertyElementMain
    )
    var uid: CFString = "" as CFString
    var dataSize = UInt32(MemoryLayout<CFString>.size)
    guard AudioObjectGetPropertyData(
      deviceID, &propAddr, 0, nil, &dataSize, &uid
    ) == noErr
    else { return "" }
    return uid as String
  }

  private func hasOutputStreams(_ deviceID: AudioDeviceID) -> Bool {
    var propAddr = AudioObjectPropertyAddress(
      mSelector: kAudioDevicePropertyStreams,
      mScope: kAudioObjectPropertyScopeOutput,
      mElement: kAudioObjectPropertyElementMain
    )
    var dataSize: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(
      deviceID, &propAddr, 0, nil, &dataSize
    ) == noErr
    else { return false }
    return dataSize > 0
  }

  private func isDeviceAlive(_ deviceID: AudioDeviceID) -> Bool {
    var propAddr = AudioObjectPropertyAddress(
      mSelector: kAudioDevicePropertyDeviceIsAlive,
      mScope: kAudioObjectPropertyScopeGlobal,
      mElement: kAudioObjectPropertyElementMain
    )
    var isAlive: UInt32 = 0
    var dataSize = UInt32(MemoryLayout<UInt32>.size)
    let status = AudioObjectGetPropertyData(deviceID, &propAddr, 0, nil, &dataSize, &isAlive)
    if status != noErr { return false }
    return isAlive != 0
  }

  private func getDefaultOutputDevice() -> AudioDeviceID {
    var propAddr = AudioObjectPropertyAddress(
      mSelector: kAudioHardwarePropertyDefaultOutputDevice,
      mScope: kAudioObjectPropertyScopeGlobal,
      mElement: kAudioObjectPropertyElementMain
    )
    var deviceID: AudioDeviceID = kAudioObjectUnknown
    var dataSize = UInt32(MemoryLayout<AudioDeviceID>.size)
    AudioObjectGetPropertyData(
      AudioObjectID(kAudioObjectSystemObject), &propAddr, 0, nil, &dataSize, &deviceID
    )
    return deviceID
  }

  private func getSampleRate(for deviceID: AudioDeviceID) -> Float64 {
    var propAddr = AudioObjectPropertyAddress(
      mSelector: kAudioDevicePropertyNominalSampleRate,
      mScope: kAudioObjectPropertyScopeGlobal,
      mElement: kAudioObjectPropertyElementMain
    )
    var sampleRate: Float64 = 48000.0
    var dataSize = UInt32(MemoryLayout<Float64>.size)
    AudioObjectGetPropertyData(deviceID, &propAddr, 0, nil, &dataSize, &sampleRate)
    return sampleRate
  }

  func getDeviceName(_ deviceID: AudioDeviceID) -> String {
    var propAddr = AudioObjectPropertyAddress(
      mSelector: kAudioObjectPropertyName,
      mScope: kAudioObjectPropertyScopeGlobal,
      mElement: kAudioObjectPropertyElementMain
    )
    var name: CFString = "" as CFString
    var dataSize = UInt32(MemoryLayout<CFString>.size)
    guard AudioObjectGetPropertyData(
      deviceID, &propAddr, 0, nil, &dataSize, &name
    ) == noErr
    else { return "Unknown" }
    return name as String
  }
}

// MARK: - Engine Device Change Callback

private let engineDeviceChangedCallback: AudioObjectPropertyListenerProc = {
  _, _, _, clientData -> OSStatus in
  guard let clientData else { return noErr }
  let engine = Unmanaged<AudioEngine>.fromOpaque(clientData).takeUnretainedValue()

  DispatchQueue.main.async {
    engine.handleDefaultDeviceChanged()
  }

  return noErr
}

private let engineDeviceListChangedCallback: AudioObjectPropertyListenerProc = {
  _, _, _, clientData -> OSStatus in
  guard let clientData else { return noErr }
  let engine = Unmanaged<AudioEngine>.fromOpaque(clientData).takeUnretainedValue()

  DispatchQueue.main.async {
    engine.handleDeviceListChanged()
  }

  return noErr
}
