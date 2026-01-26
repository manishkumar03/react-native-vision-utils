package com.visionutils

import android.content.ContentResolver
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Base64
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL

/**
 * Loads images from various sources (URL, file, base64, asset, photo library)
 */
object ImageLoader {

    /**
     * Load an image from the given source
     */
    suspend fun loadImage(context: Context, source: ImageSource): Bitmap {
        return when (source.type) {
            ImageSourceType.URL -> loadFromUrl(source.value)
            ImageSourceType.FILE -> loadFromFile(source.value)
            ImageSourceType.BASE64 -> loadFromBase64(source.value)
            ImageSourceType.ASSET -> loadFromAsset(context, source.value)
            ImageSourceType.PHOTO_LIBRARY -> loadFromPhotoLibrary(context, source.value)
        }
    }

    /**
     * Load image from URL using HttpURLConnection with redirect support
     */
    private suspend fun loadFromUrl(urlString: String): Bitmap {
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
                    val bitmap = BitmapFactory.decodeStream(inputStream)
                    inputStream.close()
                    connection.disconnect()

                    return@withContext bitmap ?: throw VisionUtilsException(
                        "LOAD_ERROR",
                        "Failed to decode image from URL: $urlString"
                    )
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
    private suspend fun loadFromFile(filePath: String): Bitmap {
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

                val options = BitmapFactory.Options().apply {
                    inPreferredConfig = Bitmap.Config.ARGB_8888
                }

                BitmapFactory.decodeFile(path, options)
                    ?: throw VisionUtilsException(
                        "LOAD_ERROR",
                        "Failed to decode image from file: $path"
                    )
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
    private suspend fun loadFromBase64(base64String: String): Bitmap {
        return withContext(Dispatchers.IO) {
            try {
                // Remove data URI prefix if present
                val cleanedBase64 = if (base64String.contains(",")) {
                    base64String.substringAfter(",")
                } else {
                    base64String
                }

                val decodedBytes = Base64.decode(cleanedBase64, Base64.DEFAULT)

                val options = BitmapFactory.Options().apply {
                    inPreferredConfig = Bitmap.Config.ARGB_8888
                }

                BitmapFactory.decodeByteArray(decodedBytes, 0, decodedBytes.size, options)
                    ?: throw VisionUtilsException(
                        "LOAD_ERROR",
                        "Failed to decode base64 image data"
                    )
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
    private suspend fun loadFromPhotoLibrary(context: Context, uriString: String): Bitmap {
        return withContext(Dispatchers.IO) {
            try {
                val uri = Uri.parse(uriString)
                val contentResolver: ContentResolver = context.contentResolver

                val inputStream = contentResolver.openInputStream(uri)
                    ?: throw VisionUtilsException(
                        "LOAD_ERROR",
                        "Failed to open input stream for URI: $uriString"
                    )

                val options = BitmapFactory.Options().apply {
                    inPreferredConfig = Bitmap.Config.ARGB_8888
                }

                val bitmap = BitmapFactory.decodeStream(inputStream, null, options)
                inputStream.close()

                bitmap ?: throw VisionUtilsException(
                    "LOAD_ERROR",
                    "Failed to decode photo library image"
                )
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
}
