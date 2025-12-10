package furhatos.app.templateadvancedskill.perception

/**
 * Holds the currently recognized user profile for the running skill instance.
 * Uses a stable temp ID when no backend identity has been resolved yet so we
 * can still persist trivia stats and name updates once known.
 */
object UserState {

    @Volatile
    var currentProfile: UserProfile? = null

    @Volatile
    private var tempUserId: String? = null

    /**
     * Merge an IdentityUpdatePayload into the local UserProfile.
     */
    fun applyIdentityUpdate(update: IdentityUpdatePayload) {
        val existing = currentProfile
        if (existing == null || existing.id != update.userId) {
            currentProfile = UserProfile(
                id = update.userId,
                name = update.name,
                primaryLanguage = update.primaryLanguage,
                languages = update.languages.toMutableMap(),
                confidence = update.confidence,
                lastSeen = update.lastSeen
            )
        } else {
            if (update.name != null) {
                existing.name = update.name
            }
            existing.primaryLanguage = update.primaryLanguage
            existing.languages.clear()
            existing.languages.putAll(update.languages)
            existing.confidence = update.confidence
            existing.lastSeen = update.lastSeen
        }
    }

    /**
     * Returns the current user id if known; otherwise returns a stable
     * session-local temporary id to allow persistence (scores, stats)
     * even when perception has not yet resolved a biometric identity.
     */
    fun getOrCreateTempId(): String {
        val existing = tempUserId
        if (existing != null) return existing
        val generated = "temp-" + java.util.UUID.randomUUID().toString()
        tempUserId = generated
        return generated
    }
}