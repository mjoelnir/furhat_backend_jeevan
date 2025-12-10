
package furhatos.app.templateadvancedskill

import java.net.InetSocketAddress
import java.net.URI
import java.net.Socket
/**
 * Central place for backend/robot URLs.
 *
 * BACKEND_URL defaults to the laptop IP (override via env BACKEND_URL).
 * ROBOT_IP_ADDRESS tracks the current robot address for reachability/debug.
 */
object params {
    private const val LAPTOP_BACKEND_URL = ""
    private const val CLOUD_BACKEND_URL = "http://51.20.7.41:8000"
    private const val LOCAL_BACKEND_URL = "http://localhost:8000"

    private val envOverride = System.getenv("BACKEND_URL")?.takeIf { it.isNotBlank() }

    /**
     * Default to the developer laptop IP unless explicitly overridden.
     * This ensures the robot never falls back to localhost, which it cannot reach.
     */
    val BACKEND_URL: String by lazy {
        val resolved = envOverride ?: LAPTOP_BACKEND_URL
        println("[params] BACKEND_URL resolved to $resolved")
        resolved
    }

    // Updated to the latest robot IP provided by the user.
    val ROBOT_IP_ADDRESS = ""

    /**
     * Kept around in case we want to re-enable automatic detection later.
     */
    @Suppress("unused")
    private fun determineBackendUrl(): String {
        return when {
            isHostReachable(LAPTOP_BACKEND_URL) -> LAPTOP_BACKEND_URL
            isHostReachable(CLOUD_BACKEND_URL) -> CLOUD_BACKEND_URL
            else -> LOCAL_BACKEND_URL
        }
    }

    private fun isHostReachable(targetUrl: String): Boolean {
        return try {
            val uri = URI(targetUrl)
            val port = if (uri.port != -1) {
                uri.port
            } else {
                when (uri.scheme?.lowercase()) {
                    "https" -> 443
                    else -> 80
                }
            }
            Socket().use { socket ->
                socket.connect(InetSocketAddress(uri.host, port), 1500)
                true
            }
        } catch (_: Exception) {
            false
        }
    }
}