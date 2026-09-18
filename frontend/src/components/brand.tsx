import Icon from "./icon";

export function WelangBrand({ href = "/" }: { href?: string }) {
  return <a className="wm-brand" href={href} aria-label="ITS Water Dashboard">
    <span className="wm-logo"><Icon name="wave" /></span>
    <span className="wm-brand-copy"><strong>ITS Water Dashboard</strong></span>
  </a>;
}

export function SiteFooter() {
  return <footer className="site-footer">
    <div className="site-footer-brand"><div><strong>Department of Civil Engineering</strong><span>Institut Teknologi Sepuluh Nopember (ITS)</span></div></div>
  </footer>;
}
