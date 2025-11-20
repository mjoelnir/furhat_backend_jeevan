package furhatos.app.templateadvancedskill.perception

object UserState {

    @Volatile
    var currentProfile: UserProfile? = null

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
}