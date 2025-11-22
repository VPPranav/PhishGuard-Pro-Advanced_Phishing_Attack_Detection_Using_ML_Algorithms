#!/bin/bash
URLS=(
"https://safe-notification-center.com/customer/authenticate/update?auth_token=7129837192837ajshdjashd8912739812739812739&continue=https://userpanel-update.com"
"https://www.google.com"
"https://roblox.mq/games/126884695634066/Grow-a-Garden?privateServerLinkCode=26438285138228421206542638963486"
"https://bank-identity-protect.com/update/confirm/account?client=17298379821739812793127398&token=ajsdhjashdjashdjashdjashd123123&return=secure"
"https://www.youtube.com"
"https://upho-ldlogx1.godaddysites.com/"
"https://amazon-secure-update.com/order/validate/payment?order=891273981273981273981273981273&securetoken=asdhjashdjashdjashdjash&redirect=https://amazon.in/orders"
"https://www.researchgate.net"
"https://www.microsoft.com"
"https://checkin-arrivals.com/YB4CMRG2H"
"https://www.amazon.in"
"https://meta-fenvor-sytron-havrel.pages.dev/"
"https://www.apple.com"
"https://appleid-restore-access.com/auth/validate/login?uid=ASD8912739812739812739812739812ASD812739&continue=https://appleid.apple.com/"
"https://www.netflix.com"
"http://gemini-usa-loggieez.godaddysites.com/"
"https://paypal-check-validation.com/secure/login/authenticate?session_id=9821739812739812739812739812739812&user=9281739182&step=verification"
"https://steamcommunity-check-protection.com/login/confirm?auth=937129837198237198273981273981273981273981&shared=9283719823719823&lang=en"
"https://stackoverflow.com"
"https://www.coursera.org"
"https://security-update-login.com/login/verify/account?session=81927398712983719283719283719283712&redirect=https://signin-security-validate.com/account/login/secure"
"https://checkin-arrivals.com/Q5RA6OLY6"
"https://social--social-8b962.us-central1.hosted.app/"
"https://45-56-104-214.cprapid.com/de/update.php"
"https://coiincobaseprolog-3in.godaddysites.com/"
"https://facebook-account-recovery.com/secure/relogin?checkpoint=892173981273981273981273981273&next=https://facebook.com/settings"
"https://www.linkedin.com"
"http://pub-b8735748a9474ef684b83941ce4cdb65.r2.dev/index.html"
"https://www.instagram.com"
"https://actualizarla-facturacion.com/"
"http://www.jovielogin.vercel.app/"
"https://www.github.com"
"https://www.wikipedia.org"

)

echo "===== MIXED URL TEST ====="
for url in "${URLS[@]}"; do
    resp=$(curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"url\": \"$url\"}")
    verdict=$(echo "$resp" | jq -r '.verdict')
    proba=$(echo "$resp" | jq -r '.probability')
    printf "\nURL: %s\n → Verdict: %s | Probability: %s\n" "$url" "$verdict" "$proba"
done
echo "===== DONE ====="

