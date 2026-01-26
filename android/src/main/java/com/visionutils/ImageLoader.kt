package com.visionutils

import android.content.ContentResolver
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Base64
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL

/**
 * Result of loading an image, includes metadata
 */
data class ImageLoadResult(
    val bitmap: Bitmap,
    val fileSize: Int?,
    val format: String?
)

/**
 * Loads images from various sources (URL, file, base64, asset, photo library)
 */
object ImageLoader {

    /**
     * Load an image from the given source
     */
    suspend fun loadImage(context: Context, source: ImageSource): Bitmap {
        return loadImageWithMetadata(context, source).bitmap
    }

    /**
     * Load an image with metadata (file size, format)
     */
    suspend fun loadImageWithMetadata(context: Context, source: ImageSource): ImageLoadResult {
        return when (source.type) {
            ImageSourceType.URL -> loadFromUrlWithMetadata(source.value)
            ImageSourceType.FILE -> loadFromFileWithMetadata(source.value)
            ImageSourceType.BASE64 -> loadFromBase64WithMetadata(source.value)
            ImageSourceType.ASSET -> {
                val bitmap = loadFromAsset(context, source.value)
                ImageLoadResult(bitmap, null, null)
            }
            ImageSourceType.PHOTO_LIBRARY -> loadFromPhotoLibraryWithMetadata(context, source.value)
        }
    }

    /**
     * Load image from URL using HttpURLConnection with redirect support
     */
    private suspend fun loadFromUrlWithMetadata(urlString: String): ImageLoadResult {
        return withContext(Dispatchers.IO) {
            try {
                var currentUrl = urlString
                var redirectCount = 0
                val maxRedirects = 5

                while (redirectCount < maxRedirects) {
                    val url = URL(currentUrl)
                    val connection = url.openConnection() as HttpURLConnection
                    connection.connectTimeout = 30000
                    connection.readTimeout = 30000
                    connection.instanceFollowRedirects = false
                    connection.setRequestProperty("User-Agent", "ReactNativeVisionUtils/1.0")
                    connection.doInput = true
                    connection.connect()

                    val responseCode = connection.responseCode

                    if (responseCode in 300..399) {
                        val redirectUrl = connection.getHeaderField("Location")
                        connection.disconnect()
                        if (redirectUrl != null) {
                            currentUrl = if (redirectUrl.startsWith("http")) {
                                redirectUrl
                            } else {
                                URL(url, redirectUrl).toString()
                            }
                            redirectCount++
                            continue
                        } else {
                            throw VisionUtilsException(
                                "LOAD_ERROR",
                                "Redirect without Location header"
                            )
                        }
                    }

                    if (responseCode != HttpURLConnection.HTTP_OK) {
                        connection.disconnect()
                        throw VisionUtilsException(
                            "LOAD_ERROR",
                            "HTTP error: $responseCode for URL: $urlString"
                        )
                    }

                    val inputStream: InputStream = connection.getInputStream()
                    val byteArrayOutputStream = ByteArrayOutputStream()
                    inputStream.copyTo(byteArrayOutputStream)
                    val bytes = byteArrayOutputStream.toByteArray()
                    inputStream.close()
                    connection.disconnect()

                    val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                        ?: throw VisionUtilsException(
                            "LOAD_ERROR",
                            "Failed to decode image from URL: $urlString"
                        )

                    val format = detectImageFormat(bytes)
                    return@withContext ImageLoadResult(bitmap, bytes.size, format)
                }

                throw VisionUtilsException(
                    "LOAD_ERROR",
                    "Too many redirects for URL: $urlString"
                )
            } catch (e: VisionUtilsException) {
                throw e
            } catch (e: Exception) {
                throw VisionUtilsException(
                    "LOAD_ERROR",
                    "Failed to load image from URL: ${e.message}"
                )
            }
        }
    }

    /**
     * Load image from local file path
     */
    private suspend fun loadFromFileWithMetadata(filePath: String): ImageLoadResult {
        return withContext(Dispatchers.IO) {
            try {
                val path = if (filePath.startsWith("file://")) {
                    filePath.removePrefix("file://")
                } else {
                    filePath
                }

                val file = File(path)
                if (!file.exists()) {
                    throw VisionUtilsException(
                        "FILE_NOT_FOUND",
                        "File not found: $path"
                    )
                }

                val bytes = file.readBytes()
                val options = BitmapFactory.Options().apply {
                    inPreferredConfig = Bitmap.Config.ARGB_8888
                }

                val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, options)
                    ?: throw VisionUtilsException(
                        "LOAD_ERROR",
                        "Failed to decode image from file: $path"
                    )

                val format = detectImageFormat(bytes)
                ImageLoadResult(bitmap, bytes.size, format)
            } catch (e: VisionUtilsException) {
                throw e
            } catch (e: Exception) {
                throw VisionUtilsException(
                    "LOAD_ERROR",
                    "Failed to load image from file: ${e.message}"
                )
            }
        }
    }

    /**
     * Load image from base64 encoded string
     */
    private suspend fun loadFromBase64WithMetadata(base64String: String): ImageLoadResult {
        return withContext(Dispatchers.IO) {
            try {
                // Extract format from data URI if present
                var format: String? = null
                val cleanedBase64 = if (base64String.contains(",")) {
                    val prefix = base64String.substringBefore(",")
                    when {
                        prefix.contains("image/jpeg") || prefix.contains("image/jpg") -> format = "jpeg"
                        prefix.contains("image/png") -> format = "png"
                        prefix.contains("image/webp") -> format = "webp"
                        prefix.contains("image/gif") -> format = "gif"
                        prefix.contains("image/heic") -> format = "heic"
                    }
                    base64String.substringAfter(",")
                } else {
                    base64String
                }

                val decodedBytes = Base64.decode(cleanedBase64, Base64.DEFAULT)

                val options = BitmapFactory.Options().apply {
                    inPreferredConfig = Bitmap.Config.ARGB_8888
                }

                val bitmap = BitmapFactory.decodeByteArray(decodedBytes, 0, decodedBytes.size, options)
                    ?: throw VisionUtilsException(
                        "LOAD_ERROR",
                        "Failed to decode base64 image data"
                    )

                // If format not detected from prefix, detect from bytes
                if (format == null) {
                    format = detectImageFormat(decodedBytes)
                }

                ImageLoadResult(bitmap, decodedBytes.size, format)
            } catch (e: VisionUtilsException) {
                throw e
            } catch (e: IllegalArgumentException) {
                throw VisionUtilsException(
                    "INVALID_SOURCE",
                    "Invalid base64 string: ${e.message}"
                )
            } catch (e: Exception) {
                throw VisionUtilsException(
                    "LOAD_ERROR",
                    "Failed to load image from base64: ${e.message}"
                )
            }
        }
    }

    /**
     * Load image from app assets
     */
    private suspend fun loadFromAsset(context: Context, assetName: String): Bitmap {
        return withContext(Dispatchers.IO) {
            try {
                val cleanAssetName = assetName
                    .removePrefix("asset://")
                    .removePrefix("assets://")
                    .removePrefix("/")

                val assetManager = context.assets
                val inputStream = assetManager.open(cleanAssetName)

                val options = BitmapFactory.Options().apply {
                    inPreferredConfig = Bitmap.Config.ARGB_8888
                }

                val bitmap = BitmapFactory.decodeStream(inputStream, null, options)
                inputStream.close()

                bitmap ?: throw VisionUtilsException(
                    "LOAD_ERROR",
                    "Failed to decode asset image: $cleanAssetName"
                )
            } catch (e: VisionUtilsException) {
                throw e
            } catch (e: java.io.FileNotFoundException) {
                throw VisionUtilsException(
                    "FILE_NOT_FOUND",
                    "Asset not found: $assetName"
                )
            } catch (e: Exception) {
                throw VisionUtilsException(
                    "LOAD_ERROR",
                    "Failed to load asset: ${e.message}"
                )
            }
        }
    }

    /**
     * Load image from photo library using content URI
     */
    private suspend fun loadFromPhotoLibraryWithMetadata(context: Context, uriString: String): ImageLoadResult {
        return withContext(Dispatchers.IO) {
            try {
                val uri = Uri.parse(uriString)
                val contentResolver: ContentResolver = context.contentResolver

                val inputStream = contentResolver.openInputStream(uri)
                    ?: throw VisionUtilsException(
                        "LOAD_ERROR",
                        "Failed to open input stream for URI: $uriString"
                    )

                val byteArrayOutputStream = ByteArrayOutputStream()
                inputStream.copyTo(byteArrayOutputStream)
                val bytes = byteArrayOutputStream.toByteArray()
                inputStream.close()

                val options = BitmapFactory.Options().apply {
                    inPreferredConfig = Bitmap.Config.ARGB_8888
                }

                val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, options)
                    ?: throw VisionUtilsException(
                        "LOAD_ERROR",
                        "Failed to decode photo library image"
                    )

                val format = detectImageFormat(bytes)
                ImageLoadResult(bitmap, bytes.size, format)
            } catch (e: VisionUtilsException) {
                throw e
            } catch (e: SecurityException) {
                throw VisionUtilsException(
                    "PERMISSION_DENIED",
                    "Permission denied to access photo: ${e.message}"
                )
            } catch (e: Exception) {
                throw VisionUtilsException(
                    "LOAD_ERROR",
                    "Failed to load from photo library: ${e.message}"
                )
            }
        }
    }

    /**
     * Detect image format from raw bytes
     */
    private fun detectImageFormat(bytes: ByteArray): String? {
        if (bytes.size < 8) return null

        // JPEG: FF D8 FF
        if (bytes[0] == 0xFF.toByte() && bytes[1] == 0xD8.toByte() && bytes[2] == 0xFF.toByte()) {
            return "jpeg"
        }

        // PNG: 89 50 4E 47 0D 0A 1A 0A
        if (bytes[0] == 0x89.toByte() && bytes[1] == 0x50.toByte() &&
            bytes[2] == 0x4E.toByte() && bytes[3] == 0x47.toByte()) {
            return "png"
        }

        // GIF: 47 49 46 38
        if (bytes[0] == 0x47.toByte() && bytes[1] == 0x49.toByte() &&
            bytes[2] == 0x46.toByte() && bytes[3] == 0x38.toByte()) {
            return "gif"
        }

        // WebP: 52 49 46 46 ... 57 45 42 50
        if (bytes.size >= 12 &&
            bytes[0] == 0x52.toByte() && bytes[1] == 0x49.toByte() &&
            bytes[2] == 0x46.toByte() && bytes[3] == 0x46.toByte() &&
            bytes[8] == 0x57.toByte() && bytes[9] == 0x45.toByte() &&
            bytes[10] == 0x42.toByte() && bytes[11] == 0x50.toByte()) {
            return "webp"
        }

        // HEIC/HEIF: Check for ftyp box
        if (bytes.size >= 12) {
            val ftypString = String(bytes.copyOfRange(4, 12), Charsets.US_ASCII)
            if (ftypString.contains("heic") || ftypString.contains("heif") || ftypString.contains("mif1")) {
                return "heic"
            }
        }

        return null
    }
}
