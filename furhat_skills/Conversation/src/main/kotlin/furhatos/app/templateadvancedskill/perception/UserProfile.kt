package furhatos.app.templateadvancedskill.perception

import com.fasterxml.jackson.annotation.JsonProperty

/**
 * Local representation of the current user as known by the backend.
 */
data class UserProfile(
    val id: String,
    var name: String? = null,
    var primaryLanguage: String = "en",
    var languages: MutableMap<String, Double> = mutableMapOf("en" to 1.0),
    var confidence: Double = 1.0,
    var lastSeen: Long = System.currentTimeMillis()
)

/**
 * Shape of the 'payload' in the 'identity_update' messages from the backend.
 * Must match the JSON sent by perception_ws_handler.py.
 */
data class IdentityUpdatePayload(
    @JsonProperty("session_id")
    val sessionId: String,

    @JsonProperty("user_id")
    val userId: String,

    val name: String?,

    @JsonProperty("primary_language")
    val primaryLanguage: String,

    val languages: Map<String, Double>,

    val confidence: Double,

    @JsonProperty("last_seen")
    val lastSeen: Long
)

/**
 * Full identity_update message from backend: { "type": "...", "payload": { ... } }
 */
data class IdentityUpdateMessage(
    val type: String,
    val payload: IdentityUpdatePayload
)