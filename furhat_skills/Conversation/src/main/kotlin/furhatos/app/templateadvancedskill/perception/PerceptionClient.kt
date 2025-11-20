package furhatos.app.templateadvancedskill.perception

import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fasterxml.jackson.module.kotlin.readValue
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import okhttp3.*
import okio.ByteString.Companion.toByteString

/**
 * Client for the /ws/perception WebSocket endpoint.
 */
class PerceptionClient(
    private val backendWsUrl: String, // e.g. "ws://localhost:8000/ws/perception"
    private val sessionId: String,
    private val robotId: String = "furhat-ntnu-01"
) {

    private val client = OkHttpClient()
    private val mapper = jacksonObjectMapper()

    @Volatile
    private var webSocket: WebSocket? = null

    /**
     * Connect to the backend WebSocket in the given coroutine scope.
     * This should be called once per Furhat interaction session.
     */
    fun connect(scope: CoroutineScope) {
        val request = Request.Builder()
            .url(backendWsUrl)
            .build()

        val listener = object : WebSocketListener() {

            override fun onOpen(ws: WebSocket, response: Response) {
                webSocket = ws
                sendHello()
            }

            override fun onMessage(ws: WebSocket, text: String) {
                handleIncomingMessage(text)
            }

            override fun onFailure(ws: WebSocket, t: Throwable, response: Response?) {
                // Log as needed; for now just print
                println("Perception WS failure: ${t.message}")
            }

            override fun onClosed(ws: WebSocket, code: Int, reason: String) {
                println("Perception WS closed: $code / $reason")
            }
        }

        scope.launch(Dispatchers.IO) {
            client.newWebSocket(request, listener)
        }
    }

    /**
     * Send "hello" message once connection opens.
     */
    private fun sendHello() {
        val msg = mapOf(
            "type" to "hello",
            "payload" to mapOf(
                "session_id" to sessionId,
                "robot_id" to robotId,
                "timestamp" to System.currentTimeMillis()
            )
        )
        sendJson(msg)
    }

    /**
     * Send a dialogue turn (user text + robot text) to the backend.
     * Can be called from onResponse / after you know what Furhat said.
     */
    fun sendTurn(
        userId: String?,
        language: String?,
        userText: String?,
        robotText: String?
    ) {
        val msg = mapOf(
            "type" to "turn",
            "payload" to mapOf(
                "session_id" to sessionId,
                "user_id" to userId,
                "language" to (language ?: "en"),
                "user_text" to (userText ?: ""),
                "robot_text" to robotText,
                "timestamp" to System.currentTimeMillis()
                // turn_index is handled by backend session_state
            )
        )
        sendJson(msg)
    }

    /**
     * Notify backend when you’ve captured the user's name.
     */
    fun sendNameUpdate(
        userId: String?,
        name: String
    ) {
        val msg = mapOf(
            "type" to "name_update",
            "payload" to mapOf(
                "session_id" to sessionId,
                "user_id" to userId,
                "name" to name,
                "timestamp" to System.currentTimeMillis()
            )
        )
        sendJson(msg)
    }

    /**
     * Close the WebSocket when the interaction ends.
     */
    fun close() {
        webSocket?.close(1000, "Session ended")
        webSocket = null
    }

    // -------------------- Internals -------------------- //

    private fun sendJson(body: Any) {
        try {
            val json: String = mapper.writeValueAsString(body)
            webSocket?.send(json)
        } catch (e: Exception) {
            println("Perception WS send error: ${e.message}")
        }
    }

    fun sendBinary(payload: ByteArray) {
        try {
            webSocket?.send(payload.toByteString())
        } catch (e: Exception) {
            println("Perception WS binary send error: ${e.message}")
        }
    }

    private fun handleIncomingMessage(text: String) {
        try {
            // We only care about identity_update for now
            val root: Map<String, Any?> = mapper.readValue(text)
            val type = root["type"] as? String ?: return
            if (type != "identity_update") return

            // Re-serialize payload to map it into IdentityUpdatePayload
            val payloadObj = root["payload"]
            val payloadJson = mapper.writeValueAsString(payloadObj)
            val payload: IdentityUpdatePayload = mapper.readValue(payloadJson)

            UserState.applyIdentityUpdate(payload)
        } catch (e: Exception) {
            println("Perception WS parse error: ${e.message}")
        }
    }
}