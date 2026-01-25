package com.visionutils

import com.facebook.react.bridge.ReactApplicationContext

class VisionUtilsModule(reactContext: ReactApplicationContext) :
  NativeVisionUtilsSpec(reactContext) {

  override fun multiply(a: Double, b: Double): Double {
    return a * b
  }

  companion object {
    const val NAME = NativeVisionUtilsSpec.NAME
  }
}
